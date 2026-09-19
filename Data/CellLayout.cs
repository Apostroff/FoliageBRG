using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace FoliageBRG
{
    [BurstCompile]
    public struct CellLayout : IDisposable
    {
        public int Count;
        public NativeArray<AABB> AABBs;
        public float2 Origin;
        public AABB GlobalBounds;
        public float MaxDistanceSqr;

        public CellLayout(float3 origin, float3 size, float cellSize, float maxDistance)
        {
            Origin = origin.xz;
            MaxDistanceSqr = maxDistance * maxDistance;
            Count = Mathf.FloorToInt(size.x / cellSize);
            AABBs = new NativeArray<AABB>(Count * Count, Allocator.Persistent);
            FillCells(ref AABBs, ref origin, cellSize, Count);
            GlobalBounds = new AABB
            {
                Center = origin + size * 0.5f,
                Extents = size * 0.5f
            };
        }

        public int CellIndex(float2 position)
        {

            float2 relativePosition = position - GlobalBounds.Min.xz;
            float2 size = GlobalBounds.Size.xz / Count;
            int2 cell = new int2(Mathf.FloorToInt(relativePosition.x / size.x), Mathf.FloorToInt(relativePosition.y / size.y));
            if (cell.x < 0 || cell.x >= Count || cell.y < 0 || cell.y >= Count)
                return -1;

            return cell.x + cell.y * Count;
        }

        public void AddBounds(int cellIndex, in PackedMatrix packedMatrix, in PrototypePrefab prototypePrefab)
        {
            float4x4 fullMatrix = packedMatrix.fullMatrix;
            AABB worldAABB = AABB.Transform(fullMatrix, prototypePrefab.AABBLocal);
            AddBounds(cellIndex, worldAABB);
        }

        public void AddBounds(int cellIndex, in AABB bounds)
        {
            if (cellIndex == -1)
                return;

            AABB cell = AABBs[cellIndex];
            if (cell.Extents.y <= float.Epsilon)
            {
                cell.Center.y = bounds.Center.y;
                cell.Extents.y = bounds.Extents.y;
            }

            cell.Encapsulate(bounds);
            AABBs[cellIndex] = cell;
            GlobalBounds.Encapsulate(bounds);
        }

        [BurstCompile]
        private static void FillCells(ref NativeArray<AABB> aabbs, ref float3 origin, float size, int count)
        {
            float halfSize = size / 2;
            float2 layoutSize = size * count;
            for (int x = 0; x < count; x++)
            for (int y = 0; y < count; y++)
            {
                int index = x + y * count;
                AABB aabb = new AABB();
                float2 centerRelative = new float2(x, y) * size + halfSize;
                float2 center = origin.xz + centerRelative;
                aabb.Center = new float3(center.x, 0, center.y);
                float2 cellExtent = math.min(new float2(halfSize), layoutSize - centerRelative);
                aabb.Extents = new float3(cellExtent.x, 0, cellExtent.y);
                aabbs[index] = aabb;
            }
        }

        public void Dispose()
        {
            if (AABBs.IsCreated)
                AABBs.Dispose();
        }
    }
}