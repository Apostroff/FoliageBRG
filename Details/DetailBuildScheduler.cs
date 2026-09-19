using System.Collections.Generic;
using System.Diagnostics;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace FoliageBRG.Details
{
    /**
     Spreads terrain detail builds over frames.
     Main-thread work (TerrainData reads, buffer uploads, bounds) runs in slices
     under a per tick time budget, the count and fill jobs run on the workers between ticks,
     and a finished build becomes visible in one step (the swap).
     Cancelling a build whose jobs are in flight completes them synchronously (a few ms).
     **/
    public class DetailBuildScheduler
    {
        private enum Stage
        {
            
            // DetailPrototypeParams for every registered prototype
            // (ComputeDetailCoverage, sizes, bounds) and the terrain parameters>
            Params,
            CoverageLayout, // GetSupportedLayers per patch
            CoverageBlocks, // GetDetailLayer per entry (Cursor = next entry)
            Heights,// GetHeights into the native heightmap
            Holes, // GetHoles into the native hole mask
            Count, // the count job runs on the workers
            Fill, // the per prototype fill jobs write the matrices and scales into the batch lists
            Upload, // one GPU buffer + BRG batch per InstanceBatch (Cursor = next batch);
            Swap,// patch bounds into the layout cells, batches into the container
        }

        private enum StepResult
        {
            Continue,
            Yield,
            Finished,
        }

        private class Build
        {
            public int LayoutId;
            public TerrainData TerrainData;
            public string Name;
            public Vector3 Origin;
            public Stage Stage;
            public int Cursor;
            public int Frames;
            public DetailSourceData Source;
            public NativeArray<int> Counts;
            public NativeArray<AABB> Bounds;
            public NativeList<InstanceBatch> Batches;
            public readonly List<GraphicsBuffer> Buffers = new();
            public JobHandle Handle;
            public int MaxInstances;
            public DetailGenerationStats Stats;
            public double UploadMs;
        }

        private readonly FoliageBRGSystem _system;
        private readonly BatchesContainer _container;
        private readonly CellLayoutManager _layouts;
        private readonly List<Build> _builds = new();

        public DetailBuildScheduler(FoliageBRGSystem system, BatchesContainer container, CellLayoutManager layouts)
        {
            _system = system;
            _container = container;
            _layouts = layouts;
        }

        public bool HasAny => _builds.Count > 0;

        public bool HasPending(int layoutId)
        {
            foreach (Build build in _builds)
            {
                if (build.LayoutId == layoutId)
                    return true;
            }

            return false;
        }

        //Queues a build for the layout, replacing a pending one
        public void Enqueue(TerrainData terrainData, int layoutId, Vector3 origin)
        {
            Cancel(layoutId);
            _builds.Add(new Build
            {
                LayoutId = layoutId,
                TerrainData = terrainData,
                Name = terrainData.name,
                Origin = origin,
                Stage = Stage.Params,
            });
        }

        public void Cancel(int layoutId)
        {
            for (int i = _builds.Count - 1; i >= 0; i--)
            {
                if (_builds[i].LayoutId != layoutId)
                    continue;

                Release(_builds[i]);
                _builds.RemoveAt(i);
            }
        }

        public void CancelAll()
        {
            foreach (Build build in _builds)
                Release(build);
            _builds.Clear();
        }

        //Advances the pending builds within <paramref name="budgetMs"/> of main-thread time. Returns true when a build was swapped in
        public bool Tick(double budgetMs)
        {
            if (_builds.Count == 0)
                return false;

            var stopwatch = Stopwatch.StartNew();
            bool swapped = false;
            for (int i = 0; i < _builds.Count;)
            {
                Build build = _builds[i];
                build.Frames++;
                if (Advance(build, stopwatch, budgetMs, ref swapped))
                    _builds.RemoveAt(i);
                else
                    i++;

                if (stopwatch.Elapsed.TotalMilliseconds >= budgetMs)
                    break;
            }

            return swapped;
        }

        // Returns true when the build is finished (swapped in or dropped)
        // and must be removed from the queue
        private bool Advance(Build build, Stopwatch stopwatch, double budgetMs, ref bool swapped)
        {
            while (stopwatch.Elapsed.TotalMilliseconds < budgetMs)
            {
                double start = stopwatch.Elapsed.TotalMilliseconds;
                StepResult result;
                switch (build.Stage)
                {
                    case Stage.Params:
                        result = StepParams(build, stopwatch, start);
                        break;
                    case Stage.CoverageLayout:
                        result = StepCoverageLayout(build, stopwatch, start);
                        break;
                    case Stage.CoverageBlocks:
                        result = StepCoverageBlocks(build, stopwatch, budgetMs, start);
                        break;
                    case Stage.Heights:
                        result = StepHeights(build, stopwatch, start);
                        break;
                    case Stage.Holes:
                        result = StepHoles(build, stopwatch, start);
                        break;
                    case Stage.Count:
                        result = StepCount(build, stopwatch, start);
                        break;
                    case Stage.Fill:
                        result = StepFill(build, stopwatch, start);
                        break;
                    case Stage.Upload:
                        result = StepUpload(build, stopwatch, budgetMs, start);
                        break;
                    case Stage.Swap:
                        result = StepSwap(build, stopwatch, start, ref swapped);
                        break;
                    default:
                        return false;
                }

                if (result == StepResult.Finished)
                    return true;
                if (result == StepResult.Yield)
                    return false;
            }

            return false;
        }

        private StepResult StepParams(Build build, Stopwatch stopwatch, double start)
        {
            if (build.TerrainData == null)
                return Drop(build);

            NativeArray<DetailPrototypeParams> prototypes = TerrainDetailReader.BuildPrototypeParams(build.TerrainData, Allocator.Persistent);
            if (prototypes.Length == 0)
            {
                prototypes.Dispose();
                return Drop(build);
            }

            build.Source = TerrainDetailReader.Create(build.TerrainData, build.Origin, prototypes);
            build.Stats.ReadMs += Since(stopwatch, start);
            build.Stage = Stage.CoverageLayout;
            return StepResult.Continue;
        }

        private StepResult StepCoverageLayout(Build build, Stopwatch stopwatch, double start)
        {
            if (build.TerrainData == null)
                return Drop(build);

            TerrainDetailReader.ReadCoverageLayout(build.TerrainData, ref build.Source, Allocator.Persistent);
            build.Stats.ReadMs += Since(stopwatch, start);
            build.Cursor = 0;
            build.Stage = Stage.CoverageBlocks;
            return StepResult.Continue;
        }

        private StepResult StepCoverageBlocks(Build build, Stopwatch stopwatch, double budgetMs, double start)
        {
            if (build.TerrainData == null)
                return Drop(build);

            int entryCount = build.Source.Entries.Length;
            while (build.Cursor < entryCount && stopwatch.Elapsed.TotalMilliseconds < budgetMs)
            {
                TerrainDetailReader.ReadCoverageBlock(build.TerrainData, ref build.Source, build.Cursor);
                build.Cursor++;
            }

            build.Stats.ReadMs += Since(stopwatch, start);
            if (build.Cursor < entryCount)
                return StepResult.Yield;

            build.Stage = Stage.Heights;
            return StepResult.Continue;
        }

        private StepResult StepHeights(Build build, Stopwatch stopwatch, double start)
        {
            if (build.TerrainData == null)
                return Drop(build);

            TerrainDetailReader.ReadHeights(build.TerrainData, ref build.Source, Allocator.Persistent);
            build.Stats.ReadMs += Since(stopwatch, start);
            build.Stage = Stage.Holes;
            return StepResult.Continue;
        }

        private StepResult StepHoles(Build build, Stopwatch stopwatch, double start)
        {
            if (build.TerrainData == null)
                return Drop(build);

            TerrainDetailReader.ReadHoles(build.TerrainData, ref build.Source, Allocator.Persistent);
            int entryCount = build.Source.Entries.Length;
            build.Counts = new NativeArray<int>(entryCount, Allocator.Persistent);
            build.Bounds = new NativeArray<AABB>(entryCount, Allocator.Persistent);
            build.Handle = TerrainDetailGenerator.ScheduleCount(build.Source, build.Counts, build.Bounds);
            JobHandle.ScheduleBatchedJobs();
            build.Stats.ReadMs += Since(stopwatch, start);
            build.Stage = Stage.Count;
            return StepResult.Yield;
        }

        private static StepResult StepCount(Build build, Stopwatch stopwatch, double start)
        {
            if (!build.Handle.IsCompleted)
                return StepResult.Yield;

            build.Handle.Complete();
            build.Batches = new NativeList<InstanceBatch>(build.Source.Prototypes.Length, Allocator.Persistent);
            build.Handle = TerrainDetailGenerator.BuildChunksAndScheduleFill(build.Source, build.Counts, build.LayoutId, build.Batches, out build.MaxInstances);
            JobHandle.ScheduleBatchedJobs();
            build.Stats.CountMs += Since(stopwatch, start);
            build.Stage = Stage.Fill;
            return StepResult.Yield;
        }

        private static StepResult StepFill(Build build, Stopwatch stopwatch, double start)
        {
            if (!build.Handle.IsCompleted)
                return StepResult.Yield;

            build.Handle.Complete();
            build.Stats.FillMs += Since(stopwatch, start);
            build.Cursor = 0;
            build.Stage = Stage.Upload;
            return StepResult.Continue;
        }

        private StepResult StepUpload(Build build, Stopwatch stopwatch, double budgetMs, double start)
        {
            int batchCount = build.Batches.Length;
            while (build.Cursor < batchCount && stopwatch.Elapsed.TotalMilliseconds < budgetMs)
            {
                ref InstanceBatch batch = ref build.Batches.ElementAt(build.Cursor);
                build.Buffers.Add(_container.RefreshBuffer(ref batch)); // creates the GPU buffer and the BRG batch, invisible until the swap
                build.Cursor++;
            }

            build.UploadMs += Since(stopwatch, start);
            if (build.Cursor < batchCount)
                return StepResult.Yield;

            build.Stage = Stage.Swap;
            return StepResult.Continue;
        }

        private StepResult StepSwap(Build build, Stopwatch stopwatch, double start, ref bool swapped)
        {
            TerrainDetailGenerator.ApplyBounds(build.Source, build.Counts, build.Bounds, _layouts, build.LayoutId, build.Origin);
            _container.AppendPreparedBatches(build.LayoutId, build.Batches, build.Buffers);
            _system.EnsureInstanceIndices(build.MaxInstances);
            build.Stats.BoundsMs += Since(stopwatch, start);
            LogBuild(build);

            // The batches and buffers belong to the container now; only the list shell and the inputs are released.
            build.Batches.Dispose();
            build.Batches = default;
            build.Buffers.Clear();
            DisposeInputs(build);
            swapped = true;
            return StepResult.Finished;
        }

        private static void LogBuild(Build build)
        {
            if (TerrainDetailGenerator.DebugCompareWithUnity && build.TerrainData != null)
                TerrainDetailGenerator.LogComparison(build.TerrainData, build.Source, build.Counts);
            if (TerrainDetailGenerator.LogTimings)
                TerrainDetailGenerator.LogTimingsLine(build.Name, build.Stats, build.UploadMs, build.Frames);
        }

        private static double Since(Stopwatch stopwatch, double start)
        {
            return stopwatch.Elapsed.TotalMilliseconds - start;
        }

        private StepResult Drop(Build build)
        {
            Release(build);
            return StepResult.Finished;
        }

        private void Release(Build build)
        {
            if (build.Stage == Stage.Count || build.Stage == Stage.Fill)
                build.Handle.Complete();

            if (build.Batches.IsCreated)
            {
                for (int i = 0; i < build.Batches.Length; i++)
                {
                    ref InstanceBatch batch = ref build.Batches.ElementAt(i);
                    GraphicsBuffer buffer = i < build.Buffers.Count ? build.Buffers[i] : null;
                    _container.ReleaseUnswappedBatch(ref batch, buffer);
                }

                build.Batches.Dispose();
                build.Batches = default;
                build.Buffers.Clear();
            }

            DisposeInputs(build);
        }

        private static void DisposeInputs(Build build)
        {
            build.Source.Dispose();
            build.Source = default;
            if (build.Counts.IsCreated) build.Counts.Dispose();
            if (build.Bounds.IsCreated) build.Bounds.Dispose();
            build.Counts = default;
            build.Bounds = default;
        }
    }
}
