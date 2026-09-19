using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using FoliageBRG.Details;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    [BurstCompile]
    public class FoliageBRGSystem : IDisposable
    {
        public static FoliageBRGSystem Instance;
        private readonly BatchesContainer _instanceBatchesContainer;
        private readonly BatchRendererGroup _batchRendererGroup;
        private readonly CellLayoutManager _cellLayoutManager;
        private readonly InstancesManager _instancesManager;
        private readonly PrototypesManager _prototypesManager;

        private readonly BatchesContainer _terrainBatchesContainer;
        private readonly DetailBuildScheduler _detailBuilds;
        private bool _enabled;

        /// <summary>Main-thread time per frame spent on pending terrain detail builds (see DetailBuildScheduler).</summary>
        public static float DetailBuildBudgetMs = 5f;
        private NativeArray<int> _instanceIndices;
        public bool Disposed;

        private FoliageBRGSystem()
        {
            _batchRendererGroup = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
            _prototypesManager = new PrototypesManager(_batchRendererGroup, BRGConstants.CameraFovDefault);
            _cellLayoutManager = new CellLayoutManager();
            _terrainBatchesContainer = new BatchesContainer(_batchRendererGroup, _cellLayoutManager);
            _instanceBatchesContainer = new BatchesContainer(_batchRendererGroup, _cellLayoutManager);
            _instancesManager = new InstancesManager(_instanceBatchesContainer, _cellLayoutManager, _prototypesManager);
            _detailBuilds = new DetailBuildScheduler(this, _terrainBatchesContainer, _cellLayoutManager);

            _instanceIndices = new NativeArray<int>(262144, Allocator.Persistent);
            FillIndices(ref _instanceIndices);
            _enabled = true;
#if UNITY_EDITOR
            UnityEditor.EditorApplication.update += EditorTick;
#endif
        }

#if UNITY_EDITOR
        /// <summary>Edit mode: BeforeRender only runs on repaints, so pending detail builds are also advanced from the editor loop.</summary>
        private void EditorTick()
        {
            if (Disposed || Application.isPlaying || !_detailBuilds.HasAny)
                return;

            if (_detailBuilds.Tick(DetailBuildBudgetMs))
                UnityEditor.SceneView.RepaintAll();
        }
#endif

        public CellLayoutManager CellLayoutManager => _cellLayoutManager;

        public BatchesContainer TerrainBatchesContainer => _terrainBatchesContainer;
        
        public InstancesManager InstancesManager => _instancesManager;
        

        public void Dispose()
        {
            if (Disposed)
                return;
            Debug.Log("BRG Dispose");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.update -= EditorTick;
#endif
            _detailBuilds.CancelAll();
            _instanceIndices.Dispose();
            _cellLayoutManager.Dispose();
            _prototypesManager.Dispose();
            _terrainBatchesContainer.Dispose();
            _instanceBatchesContainer.Dispose();
            _batchRendererGroup.Dispose();
            Disposed = true;
            _enabled = false;
        }

        public static void Initialize()
        {
            Instance ??= new FoliageBRGSystem();
        }

        public void BeforeRender()
        {
            bool dirty = _instancesManager.Update();
            if (dirty)
            {
                CleanUpCellLayouts();
            }

            _detailBuilds.Tick(DetailBuildBudgetMs);
        }

        private void CleanUpCellLayouts()
        {
            using NativeList<int> emptyLayouts = new NativeList<int>(_cellLayoutManager.UsedIndices.Length, Allocator.Temp);
            foreach (var usedIndex in _cellLayoutManager.UsedIndices)
            {
                if (_terrainBatchesContainer.HasBatches(usedIndex) || 
                    _instanceBatchesContainer.HasBatches(usedIndex) ||
                    _detailBuilds.HasPending(usedIndex))
                    continue;
                
                emptyLayouts.Add(usedIndex);
            }
            
            foreach (var emptyLayout in emptyLayouts)
            {
                CellLayoutManager.UnregisterLayout(emptyLayout);
            }
        }
        
        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
        private static void FillIndices(ref NativeArray<int> instanceIndices)
        {
            for (int i = 0; i < instanceIndices.Length; i++)
            {
                instanceIndices[i] = i;
            }
        }

        public int RegisterLayout(Vector3 origin, Vector3 size, int patchCount, float maxDistance)
        {
            return _cellLayoutManager.RegisterLayout(origin, size, size.x / patchCount, maxDistance);
        }

        public void UnregisterLayout(int cellLayoutId)
        {
            if (Disposed || cellLayoutId == -1)
                return;

            if (_terrainBatchesContainer.HasBatches(cellLayoutId) ||
                _instanceBatchesContainer.HasBatches(cellLayoutId) ||
                _detailBuilds.HasPending(cellLayoutId))
            {
                return;
            }
            
            _cellLayoutManager.UnregisterLayout(cellLayoutId);
        }

        public int RegisterPrototype(GameObject prototype, ref PrototypeSettings settings, bool reloadPrototype = false)
        {
            int id = prototype.GetInstanceID();
            _prototypesManager.RegisterPrototype(id, prototype, ref settings, out _, out _, reloadPrototype);
            return id;
        }
        
        public int RegisterPrototype(GameObject prototype, 
            ref PrototypeSettings settings, out PackedMatrix matrix, out float2 scale, bool reloadPrototype = false)
        {
            int id = prototype.GetInstanceID();
            _prototypesManager.RegisterPrototype(id, prototype, ref settings, out matrix, out scale);
            return id;
        }

        public void UnregisterPrototype(int prototypeId)
        {
            if (Disposed)
                return;

            _prototypesManager.UnregisterPrototype(prototypeId);
        }

        /// <summary>
        /// Builds the detail batches of a terrain. By default the work is spread over the next frames by the
        /// DetailBuildScheduler (DetailBuildBudgetMs of main thread per frame, jobs in between) and the grass appears
        /// when the build is complete; with TerrainDetailGenerator.Synchronous everything happens inside this call.
        /// </summary>
        public void LoadDetails(TerrainData terrainData, int layoutId, Vector3 origin)
        {
            if (TerrainDetailGenerator.Synchronous)
            {
                int maxInstances = _terrainBatchesContainer.LoadAllDetails(terrainData, layoutId, origin);
                EnsureInstanceIndices(maxInstances);
                return;
            }

            _detailBuilds.Enqueue(terrainData, layoutId, origin);
        }

        /// <summary>
        /// The identity index table is copied into the draw commands per chunk without bounds checks
        /// (PrepareDrawCallsParallelJob), so it must cover the largest batch. Only call outside of culling,
        /// i.e. from LateUpdate/BeforeRender, never from OnPerformCulling.
        /// </summary>
        public void EnsureInstanceIndices(int instanceCount)
        {
            if (instanceCount <= _instanceIndices.Length)
                return;

            int size = _instanceIndices.Length;
            while (size < instanceCount)
                size *= 2;

            _instanceIndices.Dispose();
            _instanceIndices = new NativeArray<int>(size, Allocator.Persistent);
            FillIndices(ref _instanceIndices);
        }

        public NativeList<InstanceBatch> GetInstanceBatches(int layoutId)
        {
            return _terrainBatchesContainer.GetBatches(layoutId);
        }

        public void RemoveInstance(int prototypeId, int layoutId, int instanceId)
        {
            _instancesManager.RemoveInstance(prototypeId, layoutId, instanceId);
        }

        private static readonly ProfilerMarker LayoutCullingProfiler = new("LayoutCulling");
        private static readonly ProfilerMarker CellCullingProfiler = new("CellCulling");
        private static readonly ProfilerMarker PrepareDrawCellProfiler = new("prepareDrawCall");
        
        private unsafe JobHandle OnPerformCulling(
            BatchRendererGroup rendererGroup,
            BatchCullingContext cullingContext,
            BatchCullingOutput cullingOutput,
            IntPtr userContext)
        {
            if (!_enabled)
                return new JobHandle();
            
            
            // Cull CellLayouts by frustum and distance
            float3 cameraPosition = cullingContext.lodParameters.cameraPosition;
            int layoutCellsCount = _cellLayoutManager.UsedIndices.Length;
            NativeArray<bool> visibleLayouts = new NativeArray<bool>(_cellLayoutManager.TotalCount, Allocator.TempJob, NativeArrayOptions.ClearMemory);
            VisibleLayoutsExecute(ref cullingContext, ref visibleLayouts);
            
            // Cull visible layouts cells
            NativeArray<Cell> visibleCells = new NativeArray<Cell>(layoutCellsCount * _cellLayoutManager.MaxLayoutCells, Allocator.TempJob, NativeArrayOptions.ClearMemory);
            NativeArray<int> layoutMap = new NativeArray<int>(_cellLayoutManager.TotalCount, Allocator.TempJob, NativeArrayOptions.ClearMemory);
            ParallelCellFrustumJob cellFrustumJob = new ParallelCellFrustumJob()
            {
                CullingPlanes = cullingContext.cullingPlanes,
                CullingSplits = cullingContext.cullingSplits,
                UsedLayouts = _cellLayoutManager.UsedIndices,
                CameraPosition = cameraPosition,
                CellsLayouts = _cellLayoutManager.CellLayouts,
                VisibleLayouts = visibleLayouts,
                CountPerLayout = _cellLayoutManager.MaxLayoutCells,
                //Result
                LayoutMap = layoutMap,
                VisibleCells = visibleCells,
            };

            JobHandle cellFrustumHandle = cellFrustumJob.ScheduleByRef(layoutCellsCount, 8);
                
            
            // batchCount = prototypeCount * layoutCount;
            int maxDrawCalls = CalculateMaxDrawCalls(ref visibleLayouts);
            
            NativeList<DrawCallData> drawCalls = new NativeList<DrawCallData>(maxDrawCalls, Allocator.TempJob);
            NativeArray<int> counters = new NativeArray<int>(2, Allocator.TempJob, NativeArrayOptions.ClearMemory);

            int maxJobs = _terrainBatchesContainer.BatchesByLayoutId.Count + _instanceBatchesContainer.BatchesByLayoutId.Count;
            int jobsCount = 0;
            JobHandle* drawCallHandles = stackalloc JobHandle[maxJobs];
            foreach (var pair in _terrainBatchesContainer.BatchesByLayoutId)
            {
                if (!visibleLayouts[pair.Key])
                    continue;

                PrepareDrawCallsParallelJob prepareDrawCallsJob = new PrepareDrawCallsParallelJob()
                {
                    AllLayoutCells = visibleCells,
                    LayoutMap = layoutMap,
                    CellsPreLayout = _cellLayoutManager.MaxLayoutCells,
                    InstanceBatches = pair.Value,
                    CullingSplits = cullingContext.cullingSplits,
                    CullingPlanes = cullingContext.cullingPlanes,
                    CameraPosition = cameraPosition,
                    Prototypes = _prototypesManager.Prototypes,
                    ShadowPass = cullingContext.viewType == BatchCullingViewType.Light,
                    Indices = _instanceIndices,
                    Counters = counters,
                    DrawCalls = drawCalls.AsParallelWriter()
                };
                drawCallHandles[jobsCount++] = prepareDrawCallsJob.ScheduleParallelByRef(pair.Value.Length, 8, cellFrustumHandle);
            }
            
            foreach (var pair in _instanceBatchesContainer.BatchesByLayoutId)
            {
                if (!visibleLayouts[pair.Key])
                    continue;

                PrepareDrawCallsParallelJob prepareDrawCallsJob = new PrepareDrawCallsParallelJob()
                {
                    AllLayoutCells = visibleCells,
                    LayoutMap = layoutMap,
                    CellsPreLayout = _cellLayoutManager.MaxLayoutCells,
                    InstanceBatches = pair.Value,
                    CullingSplits = cullingContext.cullingSplits,
                    CullingPlanes = cullingContext.cullingPlanes,
                    CameraPosition = cameraPosition,
                    Prototypes = _prototypesManager.Prototypes,
                    ShadowPass = cullingContext.viewType == BatchCullingViewType.Light,
                    Indices = _instanceIndices,
                    Counters = counters,
                    DrawCalls = drawCalls.AsParallelWriter()
                };
                drawCallHandles[jobsCount++] = prepareDrawCallsJob.ScheduleParallelByRef(pair.Value.Length, 8, cellFrustumHandle);
            }

            var drawCallHandleCombined = JobHandleUnsafeUtility.CombineDependencies(drawCallHandles, jobsCount);
            cullingOutput.drawCommands[0] = new BatchCullingOutputDrawCommands();
            var drawCommands = (BatchCullingOutputDrawCommands*) cullingOutput.drawCommands.GetUnsafePtr();
            var drawCommandsJob = new PopulateDrawCommandJob
            {
                OutputDrawCommands = drawCommands,
                DrawCalls = drawCalls.AsDeferredJobArray(),
                Counters = counters
            };
            var drawCommandsHandle = drawCommandsJob.ScheduleByRef(drawCallHandleCombined);

            JobHandle jobsHandle = JobHandle.CombineDependencies(drawCallHandleCombined, cellFrustumHandle);
            var jobHandles = new UnsafeList<JobHandle>(6, Allocator.Temp)
            {
                drawCommandsHandle,
                visibleLayouts.Dispose(jobsHandle),
                visibleCells.Dispose(jobsHandle),
                layoutMap.Dispose(jobsHandle),
                drawCalls.Dispose(drawCommandsHandle),
                counters.Dispose(drawCommandsHandle)
            };
            
            return JobHandleUnsafeUtility.CombineDependencies(jobHandles.Ptr, jobHandles.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void VisibleLayoutsExecute(ref BatchCullingContext cullingContext, ref NativeArray<bool> visibleLayouts)
        {
            LayoutCullingProfiler.Begin();
            float3 cameraPosition = cullingContext.lodParameters.cameraPosition;
            CullLayoutsJob cullLayoutsJob = new CullLayoutsJob
            {
                CellLayouts = _cellLayoutManager.CellLayouts,
                CameraPosition = cameraPosition,
                CullingPlanes = cullingContext.cullingPlanes,
                CullingSplits = cullingContext.cullingSplits,
                UsedIndices = _cellLayoutManager.UsedIndices,
                VisibleIndices = visibleLayouts
            };
            
            cullLayoutsJob.Execute();
            LayoutCullingProfiler.End();
        }

        private int CalculateMaxDrawCalls(ref NativeArray<bool> visibleLayouts)
        {
            int drawCallCount = CalculateMaxDrawCalls(ref visibleLayouts, 
                ref _terrainBatchesContainer.BatchesByLayoutId, ref _prototypesManager.Prototypes);
            
            int drawCallCount2 = CalculateMaxDrawCalls(ref visibleLayouts, 
                ref _instanceBatchesContainer.BatchesByLayoutId,  ref _prototypesManager.Prototypes);
            
            return drawCallCount + drawCallCount2;
        }

        
        [BurstCompile]
        private static int CalculateMaxDrawCalls(ref NativeArray<bool> visibleLayouts,
            ref NativeHashMap<int, NativeList<InstanceBatch>> batches, 
            ref NativeHashMap<int, PrototypePrefab> prototypes)
        {
            int maxDrawCalls = 0;
            foreach (var keyValuePair in batches)
            {
                int layoutId = keyValuePair.Key;
                if (!visibleLayouts[layoutId])
                    continue;

                for (int i = 0; i < keyValuePair.Value.Length; i++)
                {
                    int prototypeId = keyValuePair.Value[i].PrototypeId;
                    if (prototypes.TryGetValue(prototypeId, out PrototypePrefab prototypePrefab))
                    {
                        maxDrawCalls += prototypePrefab.LoDMeshCount;
                    }
                }
            }

            return maxDrawCalls;
        }

        public void DisposeBatchesByLayoutId(int layoutId)
        {
            _detailBuilds.Cancel(layoutId);
            _terrainBatchesContainer.DisposeByLayoutId(layoutId);
        }

        public bool GetPrototype(int id, out PrototypePrefab  prototypePrefab)
        {
            return _prototypesManager.Prototypes.TryGetValue(id, out prototypePrefab);
        }

        public Bounds GetPrototypeBounds(int id)
        {
            if (_prototypesManager.Prototypes.TryGetValue(id, out var prefab))
            {
                return new Bounds(prefab.AABBLocal.Center,  prefab.AABBLocal.Size);
            }

            return default;
        }
    }
}