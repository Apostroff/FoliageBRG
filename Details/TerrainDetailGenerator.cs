using System.Diagnostics;
using System.Text;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;

namespace FoliageBRG.Details
{
    
    // Builds the detail InstanceBatches of one terrain from its coverage maps with Burst jobs,
    // replacing the per patch ComputeDetailInstanceTransforms + GetInterpolatedNormal main-thread path.
    // does everything inside one call; DetailBuildScheduler uses the same steps spread over frames.
    public static class TerrainDetailGenerator
    {
        // Logs our instance counts next to Unity's ComputeDetailInstanceTransforms counts for every prototype
        public static bool DebugCompareWithUnity;

        //One console line per terrain with the main-thread time of every phase
        public static bool LogTimings = true;

        //Build everything inside FoliageBRGSystem.LoadDetails instead of spreading it over frames
        public static bool Synchronous;

        private static readonly ProfilerMarker ReadMarker = new("TerrainDetails.Read");
        private static readonly ProfilerMarker CountMarker = new("TerrainDetails.Count");
        private static readonly ProfilerMarker FillMarker = new("TerrainDetails.Fill");
        private static readonly ProfilerMarker BoundsMarker = new("TerrainDetails.Bounds");

        
        // Synchronous build: adds one InstanceBatch per registered prototype to <paramref name="batches"/> and the patch
        // bounds to the layout. Returns the largest instance count of a single batch.
        
        public static int Generate(TerrainData terrainData, int layoutId, Vector3 origin, CellLayoutManager layouts, NativeList<InstanceBatch> batches,
            out DetailGenerationStats stats)
        {
            stats = default;
            var stopwatch = Stopwatch.StartNew();

            DetailSourceData source;
            using (ReadMarker.Auto())
            {
                NativeArray<DetailPrototypeParams> prototypes = TerrainDetailReader.BuildPrototypeParams(terrainData, Allocator.TempJob);
                if (prototypes.Length == 0)
                {
                    prototypes.Dispose();
                    stats.ReadMs = stopwatch.Elapsed.TotalMilliseconds;
                    return 0;
                }

                source = TerrainDetailReader.Read(terrainData, origin, prototypes, Allocator.TempJob);
            }

            stats.ReadMs = stopwatch.Elapsed.TotalMilliseconds;

            int entryCount = source.Entries.Length;
            NativeArray<int> counts = default;
            NativeArray<AABB> bounds = default;
            int maxInstances = 0;

            try
            {
                counts = new NativeArray<int>(entryCount, Allocator.TempJob);
                bounds = new NativeArray<AABB>(entryCount, Allocator.TempJob);

                using (CountMarker.Auto())
                {
                    ScheduleCount(source, counts, bounds).Complete();
                }

                stats.CountMs = stopwatch.Elapsed.TotalMilliseconds - stats.ReadMs;

                using (FillMarker.Auto())
                {
                    BuildChunksAndScheduleFill(source, counts, layoutId, batches, out maxInstances).Complete();
                }

                stats.FillMs = stopwatch.Elapsed.TotalMilliseconds - stats.ReadMs - stats.CountMs;

                using (BoundsMarker.Auto())
                {
                    ApplyBounds(source, counts, bounds, layouts, layoutId, origin);
                }

                stats.BoundsMs = stopwatch.Elapsed.TotalMilliseconds - stats.ReadMs - stats.CountMs - stats.FillMs;

                if (DebugCompareWithUnity)
                    LogComparison(terrainData, source, counts);
            }
            finally
            {
                if (counts.IsCreated) counts.Dispose();
                if (bounds.IsCreated) bounds.Dispose();
                source.Dispose();
            }

            return maxInstances;
        }

        // Counts the instances of every entry and computes the terrain-local patch bounds.
        // Returns a default handle when there are no entries
        public static JobHandle ScheduleCount(in DetailSourceData source, NativeArray<int> counts, NativeArray<AABB> bounds)
        {
            int entryCount = source.Entries.Length;
            if (entryCount == 0)
                return default;

            return new CountDetailInstancesJob
            {
                Entries = source.Entries,
                Prototypes = source.Prototypes,
                Coverage = source.Coverage,
                Heights = source.Heights,
                Holes = source.Holes,
                Terrain = source.Terrain,
                Counts = counts,
                Bounds = bounds,
            }.Schedule(entryCount, 4);
        }

        // Main thread: creates one InstanceBatch per prototype (appended to <paramref name="batches"/>) with its chunks and
        // offsets from <paramref name="counts"/>, then schedules the fill jobs. Entries get their InstanceStart/InstanceCount here.
        
        public static JobHandle BuildChunksAndScheduleFill(in DetailSourceData source, NativeArray<int> counts, int layoutId,
            NativeList<InstanceBatch> batches, out int maxInstances)
        {
            int slots = source.Prototypes.Length;
            NativeArray<DetailPatchEntry> entries = source.Entries;
            int batchesStart = batches.Length;
            maxInstances = 0;

            // Chunks and offsets for every prototype first: the fill jobs read Entries, so it must not change after scheduling.
            for (int s = 0; s < slots; s++)
            {
                int start = source.EntryStartPerPrototype[s];
                int end = source.EntryStartPerPrototype[s + 1];
                InstanceBatch batch = new InstanceBatch(source.Prototypes[s].RootId, source.Terrain.PatchCount, layoutId);

                int running = 0;
                for (int e = start; e < end; e++)
                {
                    int count = counts[e];
                    DetailPatchEntry entry = entries[e];
                    entry.InstanceStart = running;
                    entry.InstanceCount = count;
                    entries[e] = entry;
                    if (count == 0)
                        continue;

                    batch.InstancesChunks.Add(new InstancesChunk(running, (uint) count, entry.CellIndex));
                    running += count;
                }

                batch.InstanceCount = running;
                if (running > 0)
                {
                    batch.Transforms.ResizeUninitialized(running);
                    batch.Scales.ResizeUninitialized(running);
                }

                maxInstances = math.max(maxInstances, running);
                batches.Add(batch);
            }

            var handles = new NativeArray<JobHandle>(slots, Allocator.Temp);
            for (int s = 0; s < slots; s++)
            {
                int start = source.EntryStartPerPrototype[s];
                int end = source.EntryStartPerPrototype[s + 1];
                ref InstanceBatch batch = ref batches.ElementAt(batchesStart + s);
                if (batch.InstanceCount == 0)
                    continue;

                handles[s] = new FillDetailInstancesJob
                {
                    Entries = entries.GetSubArray(start, end - start),
                    Prototypes = source.Prototypes,
                    Coverage = source.Coverage,
                    Heights = source.Heights,
                    Holes = source.Holes,
                    Terrain = source.Terrain,
                    Transforms = batch.Transforms.AsArray(),
                    Scales = batch.Scales.AsArray(),
                }.Schedule(end - start, 2);
            }

            JobHandle combined = JobHandle.CombineDependencies(handles);
            handles.Dispose();
            return combined;
        }

        public static void ApplyBounds(in DetailSourceData source, NativeArray<int> counts, NativeArray<AABB> bounds,
            CellLayoutManager layouts, int layoutId, Vector3 origin)
        {
            float3 originOffset = origin;
            for (int e = 0; e < counts.Length; e++)
            {
                if (counts[e] == 0)
                    continue;

                AABB patchBounds = bounds[e];
                patchBounds.Center += originOffset;
                layouts.AddBounds(layoutId, source.Entries[e].CellIndex, patchBounds);
            }
        }

        public static void LogTimingsLine(string terrainName, in DetailGenerationStats stats, double uploadMs, int frames)
        {
            double total = stats.ReadMs + stats.CountMs + stats.FillMs + stats.BoundsMs + uploadMs;
            string spread = frames > 1 ? $" over {frames} frames" : string.Empty;
            UnityEngine.Debug.Log(
                $"[TerrainDetails] {terrainName}: main thread {total:F1} ms{spread} = read {stats.ReadMs:F1} + count {stats.CountMs:F1} + fill {stats.FillMs:F1} + bounds {stats.BoundsMs:F1} + upload {uploadMs:F1}");
        }

        public static void LogComparison(TerrainData terrainData, in DetailSourceData source, NativeArray<int> counts)
        {
            DetailPrototype[] prototypes = terrainData.detailPrototypes;
            int patchCount = source.Terrain.PatchCount;
            var sb = new StringBuilder();
            long oursTotal = 0;
            long unityTotal = 0;

            for (int s = 0; s < source.Prototypes.Length; s++)
            {
                int layer = source.Prototypes[s].PrototypeIndex;
                long ours = 0;
                for (int e = source.EntryStartPerPrototype[s]; e < source.EntryStartPerPrototype[s + 1]; e++)
                    ours += counts[e];

                long unity = 0;
                for (int py = 0; py < patchCount; py++)
                for (int px = 0; px < patchCount; px++)
                    unity += terrainData.ComputeDetailInstanceTransforms(px, py, layer, 1f, out _).Length;

                oursTotal += ours;
                unityTotal += unity;
                string name = prototypes[layer].prototype != null ? prototypes[layer].prototype.name : "null";
                sb.AppendLine($"  [{layer}] {name}: ours={ours} unity={unity} ratio={(unity > 0 ? (double) ours / unity : 0):F3}");
            }

            UnityEngine.Debug.Log($"[TerrainDetails] {terrainData.name}: ours={oursTotal} unity={unityTotal} ratio={(unityTotal > 0 ? (double) oursTotal / unityTotal : 0):F3}\n{sb}");
        }
    }
}
