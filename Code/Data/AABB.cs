using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace FoliageBRG
{
    [StructLayout(LayoutKind.Sequential)]
    public struct AABB
    {
        public float3 Center;
        public float3 Extents;

        public readonly float3 Min => Center - Extents;
        public readonly float3 Max => Center + Extents;
        public readonly float3 Size => Extents * 2f;

        public static AABB Transform(in float4x4 matrix, in AABB bounds)
        {
            float3 extents = bounds.Extents;

            return new AABB
            {
                Center = math.transform(matrix, bounds.Center),
                Extents = math.abs(matrix.c0.xyz) * extents.x
                          + math.abs(matrix.c1.xyz) * extents.y
                          + math.abs(matrix.c2.xyz) * extents.z
            };
        }

        public readonly override string ToString()
        {
            return $"AABB(Center: Min: {Min}, Max: {Max})";
        }
    }
}
