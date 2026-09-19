using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    [BurstCompile]
    public struct CullLayoutsJob
    {
        [ReadOnly] public NativeArray<CellLayout> CellLayouts;

        [ReadOnly] public NativeList<int> UsedIndices;

        [ReadOnly] public NativeArray<CullingSplit> CullingSplits;

        [ReadOnly] public float3 CameraPosition;

        [ReadOnly] public NativeArray<Plane> CullingPlanes;

        [WriteOnly] public NativeArray<bool> VisibleIndices;



        public void Execute()
        {
            for (int i = 0; i < UsedIndices.Length; i++)
            {
                int index = UsedIndices[i];
                CellLayout layout = CellLayouts[index];
                ref AABB aabb = ref layout.GlobalBounds;
                bool visible = BRGUtils.InFrustum(ref aabb, ref CullingPlanes, ref CullingSplits);
                if (!visible)
                {
                    VisibleIndices[index] = false;
                    continue;
                }

                float3 center = math.abs(CameraPosition - aabb.Center);
                float3 distance = math.max(center - aabb.Extents, 0);

                float distanceSqr = math.dot(distance, distance);
                VisibleIndices[index] = distanceSqr < layout.MaxDistanceSqr;
            }
        }
    }
}