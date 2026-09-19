
using System.Runtime.CompilerServices;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
    public struct PrepareDrawCallsParallelJob : IJobFor
    {
        [ReadOnly]
        public NativeArray<Cell> AllLayoutCells;
        
        [ReadOnly]
        public NativeArray<int> LayoutMap;

        [ReadOnly] 
        public int CellsPreLayout;
        
        [ReadOnly, NativeDisableContainerSafetyRestriction]
        public NativeList<InstanceBatch> InstanceBatches;
        
        [ReadOnly] 
        public NativeArray<CullingSplit> CullingSplits;
        
        [ReadOnly] 
        public NativeArray<Plane> CullingPlanes;
        
        [ReadOnly] 
        public float3 CameraPosition;
        
        [ReadOnly, NativeDisableContainerSafetyRestriction]
        public NativeHashMap<int, PrototypePrefab> Prototypes;

        [ReadOnly]
        public bool ShadowPass;
        
        [ReadOnly]
        public NativeArray<int> Indices;

        [WriteOnly, NativeDisableContainerSafetyRestriction]
        public NativeArray<int> Counters; // 0 - draw call count, 1 - instance count
        
        [WriteOnly, NativeDisableContainerSafetyRestriction]
        public NativeList<DrawCallData>.ParallelWriter DrawCalls;
        
        public void Execute(int index)
        {
            InstanceBatch instanceBatch =  InstanceBatches[index];
            if (!Prototypes.TryGetValue(instanceBatch.PrototypeId, out PrototypePrefab prototype)) 
                return;
            
            if (ShadowPass && prototype.Settings.MaxShadowDistance == 0)
                return;

            int layoutIndex = LayoutMap[instanceBatch.Layout];
            
            NativeArray<Cell> layoutCells = AllLayoutCells.GetSubArray(layoutIndex * CellsPreLayout, CellsPreLayout);
            NativeList<int2> visibleIndices;
            NativeArray<int> countsPerLod = new NativeArray<int>(prototype.LODs.Length, Allocator.Temp);
            if (prototype.Settings.PerInstanceCulling)
            {
                visibleIndices = new NativeList<int2>(instanceBatch.InstanceCount, Allocator.Temp);
                VisibleIndices(ref prototype, ref instanceBatch.InstancesChunks, ref instanceBatch.Transforms, 
                    ref countsPerLod, ref visibleIndices, ref layoutCells);
            }
            else
            {
                visibleIndices = new NativeList<int2>(instanceBatch.InstancesChunks.Length, Allocator.Temp);
                VisibleChunks(ref prototype, ref instanceBatch.InstancesChunks, ref countsPerLod, ref visibleIndices, ref layoutCells);
            }

            PrepareDrawCalls(ref visibleIndices, ref prototype, ref countsPerLod, ref instanceBatch.InstancesChunks, instanceBatch.BatchID);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void PrepareDrawCalls(ref NativeList<int2> visibleChunks, 
            ref PrototypePrefab prototype, 
            ref NativeArray<int> countsPerLod, ref NativeList<InstancesChunk> chunks, BatchID batchID)
        {
            if (visibleChunks.Length == 0)
                return;
            
            ref NativeArray<PrototypeLOD> loDs = ref prototype.LODs; 
            // Calculate the total visible count and total draw call count. 

            uint visibleInstancesCount = 0;
            int visibleDrawCallsCount = 0;
            int** lodsIndices = stackalloc int*[loDs.Length];
            for (int i = 0; i < loDs.Length; i++)
            {
                if (countsPerLod[i] == 0)
                {
                    continue;
                }

                visibleInstancesCount += (uint) countsPerLod[i];
                visibleDrawCallsCount += loDs[i].Meshes.Length;
                // Allocate memory for indices for every LOD
                lodsIndices[i] = (int*) UnsafeUtility.MallocTracked(countsPerLod[i] * sizeof(int),
                    UnsafeUtility.AlignOf<int>(), Allocator.TempJob, 0);
            }

            // Fill indices by lods 
            uint* currentInstanceIndices = stackalloc uint[loDs.Length];
            UnsafeUtility.MemClear(currentInstanceIndices, sizeof(uint) * loDs.Length);
            int* indicesPtr = (int*) Indices.GetUnsafeReadOnlyPtr();
            for (int i = 0; i < visibleChunks.Length; i++)
            {
                int2 instance = visibleChunks[i];
                if (prototype.Settings.PerInstanceCulling)
                {
                    lodsIndices[instance.y][currentInstanceIndices[instance.y]] = instance.x;
                    currentInstanceIndices[instance.y]++;
                }
                else
                {
                    InstancesChunk chunk = chunks[visibleChunks[i].x];
                    int lodIndex = visibleChunks[i].y;
                    int* indices = lodsIndices[lodIndex];
                    UnsafeUtility.MemCpy(indices + currentInstanceIndices[lodIndex], indicesPtr + chunk.IndicesStart, chunk.IndicesCount * sizeof(int));
                    currentInstanceIndices[lodIndex] += chunk.IndicesCount;
                }
            }

            // Fill draw call 
            NativeList<DrawCallData> drawCalls = new NativeList<DrawCallData>(visibleDrawCallsCount, Allocator.Temp);
            for (int i = 0; i < loDs.Length; i++)
            {
                if (countsPerLod[i] == 0)
                {
                    continue;
                }

                var lod = loDs[i];
                int meshesCount = lod.Meshes.Length;

                for (int j = 0; j < meshesCount; j++)
                {
                    MeshLOD mesh = lod.Meshes[j];
                    DrawCallData data = new DrawCallData(batchID, mesh.MaterialID, mesh.MeshID, (uint) countsPerLod[i], mesh.SubmeshIndex);
                    data.Indices = j == 0 ? lodsIndices[i] : null;
                    drawCalls.AddNoResize(data);
                }
            }

            DrawCalls.AddRangeNoResize(drawCalls);

            ref int drawCallCountRef = ref UnsafeUtility.ArrayElementAsRef<int>(Counters.GetUnsafePtr(), 0);
            ref int visibleCountRef = ref UnsafeUtility.ArrayElementAsRef<int>(Counters.GetUnsafePtr(), 1);

            Interlocked.Add(ref drawCallCountRef, visibleDrawCallsCount);
            Interlocked.Add(ref visibleCountRef, (int) visibleInstancesCount);

        }
         
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void VisibleIndices(ref PrototypePrefab prototype, 
            ref NativeList<InstancesChunk> chunks, 
            ref NativeList<PackedMatrix> transforms,
            ref NativeArray<int> countsPerLod,
            ref NativeList<int2> visibleIndices, 
            ref NativeArray<Cell> cells)
        {
            ref NativeArray<float> distances = ref prototype.Distances;
            int distancesCount = distances.Length;
            
            int* counts = stackalloc int[distancesCount];
            UnsafeUtility.MemClear(counts, sizeof(uint) * distancesCount);

            float maxDistance = distances[^1];
            float minDistanceSqr = maxDistance * maxDistance;
            
            for (int i = 0; i < chunks.Length; i++)
            {
                InstancesChunk chunk = chunks[i];
                Cell cell = cells[chunk.CellIndex];
                if (!cell.Visible || cell.Distance > maxDistance)
                    continue;

                for (int j = chunk.IndicesStart; j < chunk.IndicesStart + chunk.IndicesCount; j++)
                {
                    PackedMatrix matrix = transforms[j];
                    float3 distance = CameraPosition - matrix.GetPosition();
                    float distanceSqr = math.dot(distance, distance);
                    if (distanceSqr > minDistanceSqr)
                    {
                        continue;
                    }

                    AABB aabb = AABB.Transform(matrix.fullMatrix, prototype.AABBLocal);
                    bool visible = BRGUtils.InFrustum(ref aabb, ref CullingPlanes, ref CullingSplits);
                    if (!visible)
                        continue;

                    int lodIndex = distances.Length - 1;
                    for (int li = 0; li < distances.Length - 1; li++)
                    {
                        if (distanceSqr <= distances[li] * distances[li])
                        {
                            lodIndex = li;
                            break;
                        }
                    }

                    counts[lodIndex]++;
                    visibleIndices.AddNoResize(new int2(j, lodIndex));
                }

            }

            for (int i = 0; i < countsPerLod.Length; i++)
            {
                countsPerLod[i] = counts[i];
            }
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe void VisibleChunks(ref PrototypePrefab prototype, 
            ref  NativeList<InstancesChunk> chunks,
            ref NativeArray<int> countsPerLod,
            ref NativeList<int2> visibleIndices,
            ref NativeArray<Cell> cells)
        {
            ref NativeArray<float> lodDistances = ref prototype.Distances;
            
            int* counts = stackalloc int[lodDistances.Length];
            UnsafeUtility.MemClear(counts, sizeof(uint) * lodDistances.Length);

            for (int i = 0; i < chunks.Length; i++)
            {
                InstancesChunk chunk = chunks[i];
                int cellNo = chunk.CellIndex;

                Cell cell = cells[cellNo];
                if (!cell.Visible)
                    continue;

                if (cell.Distance > prototype.Settings.MaxDistance) // Shadows
                    continue;


                int lodIndex = lodDistances.Length - 1;
                for (int li = 0; li < lodDistances.Length - 1; li++)
                {
                    if (cell.Distance <= lodDistances[li])
                    {
                        lodIndex = li;
                        break;
                    }
                }

                visibleIndices.AddNoResize(new int2(i, lodIndex));
                counts[lodIndex] += (int) chunk.IndicesCount;
            }

            for (int i = 0; i < countsPerLod.Length; i++)
            {
                countsPerLod[i] = counts[i];
            }
        }
    }
}