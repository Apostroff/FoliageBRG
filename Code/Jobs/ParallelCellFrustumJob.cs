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
    public struct ParallelCellFrustumJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Plane> CullingPlanes;
        [ReadOnly] public NativeArray<CullingSplit> CullingSplits;
        [ReadOnly] public NativeList<int> UsedLayouts;
        
        [ReadOnly] public float3 CameraPosition;
        [ReadOnly, NativeDisableContainerSafetyRestriction] public NativeArray<CellLayout> CellsLayouts;
        [ReadOnly] public NativeArray<bool> VisibleLayouts;
        [ReadOnly]
        public int CountPerLayout;
        
        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<int> LayoutMap;
        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<Cell> VisibleCells;
        
        public void Execute(int index)
        {
            int layoutId = UsedLayouts[index];
            if (!VisibleLayouts[layoutId])
                return;

            LayoutMap[layoutId] = index;
            NativeArray<AABB> aabbs = CellsLayouts[layoutId].AABBs;
            int offset = index * CountPerLayout;
            for (int i = 0; i < aabbs.Length; i++)
            {
                Cell cell = new Cell();
                AABB aabb = aabbs[i];
                cell.Visible = BRGUtils.InFrustum(ref aabb, ref CullingPlanes, ref CullingSplits);
                if (cell.Visible)
                {
                    float3 center = math.abs(CameraPosition - aabb.Center);
                    float3 distance = math.max(center - aabb.Extents, 0);

                    cell.Distance = math.sqrt(math.dot(distance, distance));
                }

                VisibleCells[offset + i] = cell;
            }
            
        }
    }
}