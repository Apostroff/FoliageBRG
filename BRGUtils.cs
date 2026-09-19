using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    [BurstCompile]
    public static class BRGUtils
    {
        public static GameObject GetRoot(GameObject go)
        {
            if (go.transform.parent == null)
                return go;

            return GetRoot(go.transform.parent.gameObject);
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool InFrustum(ref AABB aabb, ref NativeArray<Plane> planes, int planesOffset, int planesCount)
        {
            for (var i = planesOffset; i < planesOffset + planesCount; i++)
            {
                var plane = planes[i];
                var normal = plane.normal;
                var distance = math.dot(normal, aabb.Center) + plane.distance;
                var radius = math.dot(aabb.Extents, math.abs(normal));

                if (distance + radius <= 0)
                    return false;
            }

            return true;
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool InFrustum(ref AABB aabb, ref NativeArray<Plane> planes, ref NativeArray<CullingSplit> spits)
        {
            for (var i = 0; i < spits.Length; i++)
            {
                CullingSplit split = spits[i];
                bool inFrustum = InFrustum(ref aabb, ref planes, split.cullingPlaneOffset, split.cullingPlaneCount);
                if (inFrustum)
                    return true;

            }

            return false;
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Encapsulate(ref this AABB a, in AABB b)
        {
            float3 min = math.min(a.Min, b.Min);
            float3 max = math.max(a.Max, b.Max);

            a.Center = (min + max) * 0.5f;
            a.Extents = (max - min) * 0.5f;
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Encapsulate(ref AABB aabb, ref float3 point)
        {
            float3 min = math.min(point, aabb.Min);
            float3 max = math.max(point, aabb.Max);
            aabb.Center = (max + min) * 0.5f;
            aabb.Extents = (max - min) * 0.5f;
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
        public static void FromBounds(ref this AABB aabb, in Bounds bounds)
        {
            aabb.Center = bounds.center;
            aabb.Extents = bounds.extents;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Bounds ToWorld(this Bounds bound, float4x4 matrix)
        {
            AABB aabb = new AABB
            {
                Center = bound.center,
                Extents = bound.extents
            };
            AABB world = AABB.Transform(matrix, aabb);
            return new Bounds(world.Center, world.Size);
        }
    }
}
