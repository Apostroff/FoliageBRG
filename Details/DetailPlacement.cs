using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace FoliageBRG.Details
{
    // Scatter math shared by the count and fill jobs. Reproduces the statistics of Unity's
    // ComputeDetailInstanceTransforms (legacy detail distribution, coverage mode) measured on the project terrains:
    // per cell instances = nx * nz with nx, nz = floor(n + u), n = cellSize * coverage / 255 * ComputeDetailCoverage;
    // positions uniform inside the cell, rotation uniform, one Perlin driven scalar for width and height,
    // height by triangle interpolation on the 00-11 diagonal (matches GetInterpolatedHeight and the terrain mesh).
    // Everything comes from one random stream per cell, so the count pass and the fill pass stay in lockstep.
    
    [BurstCompile]
    public static class DetailPlacement
    {
        private const float kTwoPi = 2f * math.PI;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Random CellRandom(int cellX, int cellZ, int layer, int seed)
        {
            uint hash = math.hash(new int4(cellX, cellZ, layer, seed));
            return new Random(hash == 0 ? 1u : hash);
        }

        /// Per axis count with stochastic rounding: floor(n + u).
        /// Its mean is exactly n for every n, which is what
        /// Unity's counts show down to the sparsest coverage
        /// (a symmetric +-1 jitter with a clamp at zero
        /// overshoots badly for n below 0.5, i.e. for pebbles, branches and other sparse prototypes).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int AxisCount(float expected, ref Random rng)
        {
            return (int) math.floor(expected + rng.NextFloat());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int CellInstanceCount(in DetailTerrainParams terrain, in DetailPrototypeParams prototype, byte coverage, ref Random rng)
        {
            if (terrain.InstanceCountMode)
                return coverage;

            float perMeter = coverage * (1f / 255f) * prototype.CoveragePerMeter;
            int nx = AxisCount(perMeter * terrain.CellSizeX, ref rng);
            int nz = AxisCount(perMeter * terrain.CellSizeZ, ref rng);
            return nx * nz;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float RandomRotation(ref Random rng)
        {
            return rng.NextFloat() * kTwoPi;
        }

        // Height in metres at a terrain-local XZ position, same triangle split as the terrain mesh (00-11 diagonal)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SampleHeight(in NativeArray<float> heights, int resolution, in float3 size, float localX, float localZ)
        {
            SampleCoords(resolution, size, localX, localZ, out int ix, out int iz, out float tx, out float tz);
            int row0 = iz * resolution;
            int row1 = row0 + resolution;
            float h00 = heights[row0 + ix];
            float h10 = heights[row0 + ix + 1];
            float h01 = heights[row1 + ix];
            float h11 = heights[row1 + ix + 1];
            float h = tx >= tz
                ? h00 + (h10 - h00) * tx + (h11 - h10) * tz
                : h00 + (h01 - h00) * tz + (h11 - h01) * tx;
            return h * size.y;
        }

        // Bilinear blend of vertex normals from central differences, like TerrainData.GetInterpolatedNormal
        public static float3 SampleNormal(in NativeArray<float> heights, int resolution, in float3 size, float localX, float localZ)
        {
            SampleCoords(resolution, size, localX, localZ, out int ix, out int iz, out float tx, out float tz);
            float3 n00 = VertexNormal(heights, resolution, size, ix, iz);
            float3 n10 = VertexNormal(heights, resolution, size, ix + 1, iz);
            float3 n01 = VertexNormal(heights, resolution, size, ix, iz + 1);
            float3 n11 = VertexNormal(heights, resolution, size, ix + 1, iz + 1);
            float3 n = math.lerp(math.lerp(n00, n10, tx), math.lerp(n01, n11, tx), tz);
            return math.normalizesafe(n, new float3(0, 1, 0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SampleCoords(int resolution, in float3 size, float localX, float localZ,
            out int ix, out int iz, out float tx, out float tz)
        {
            float fx = math.saturate(localX / size.x) * (resolution - 1);
            float fz = math.saturate(localZ / size.z) * (resolution - 1);
            ix = math.min((int) fx, resolution - 2);
            iz = math.min((int) fz, resolution - 2);
            tx = fx - ix;
            tz = fz - iz;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float3 VertexNormal(in NativeArray<float> heights, int resolution, in float3 size, int x, int z)
        {
            int xm = math.max(x - 1, 0);
            int xp = math.min(x + 1, resolution - 1);
            int zm = math.max(z - 1, 0);
            int zp = math.min(z + 1, resolution - 1);
            float stepX = size.x / (resolution - 1);
            float stepZ = size.z / (resolution - 1);
            float dhdx = (heights[z * resolution + xp] - heights[z * resolution + xm]) * size.y / ((xp - xm) * stepX);
            float dhdz = (heights[zp * resolution + x] - heights[zm * resolution + x]) * size.y / ((zp - zm) * stepZ);
            return math.normalize(new float3(-dhdx, 1f, -dhdz));
        }

        // True when the point's texel, or any texel within +-radius (a square, conservative), is a hole
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsHole(in NativeArray<byte> holes, int resolution, in float3 size, float localX, float localZ, float radius)
        {
            if (resolution == 0)
                return false;

            if (radius <= 0f)
                return holes[TexelIndex(localZ, size.z, resolution) * resolution + TexelIndex(localX, size.x, resolution)] == 0;

            int x0 = TexelIndex(localX - radius, size.x, resolution);
            int x1 = TexelIndex(localX + radius, size.x, resolution);
            int z0 = TexelIndex(localZ - radius, size.z, resolution);
            int z1 = TexelIndex(localZ + radius, size.z, resolution);
            for (int z = z0; z <= z1; z++)
            {
                int row = z * resolution;
                for (int x = x0; x <= x1; x++)
                {
                    if (holes[row + x] == 0)
                        return true;
                }
            }

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int TexelIndex(float local, float size, int resolution)
        {
            return math.clamp((int) (local / size * resolution), 0, resolution - 1);
        }

        // One Perlin driven scalar in [0..1] shared by width and height; spatial frequency = noiseSpread
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ScaleNoise(in DetailPrototypeParams prototype, float localX, float localZ)
        {
            float2 p = new float2(localX, localZ) * prototype.NoiseSpread + prototype.NoiseSeed * 0.7317f;
            float n = noise.cnoise(p); // roughly [-1, 1] and bell shaped, like Unity's scale distribution
            return math.saturate(0.5f + 0.6f * n);
        }

        // Same layout as the old InstanceBatch.FillMatrices: rotation around Y times scale, optional ground alignment
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PackedMatrix BuildMatrix(in float3 worldPosition, float rotationY, float scaleXZ, float scaleY, bool align, in float3 up)
        {
            math.sincos(rotationY, out float sinY, out float cosY);
            float3 c0, c1, c2;
            if (align)
            {
                float3 forward = new float3(sinY, 0, cosY);
                float3 right = math.normalizesafe(math.cross(up, forward), new float3(cosY, 0, -sinY));
                forward = math.cross(right, up);
                c0 = right * scaleXZ;
                c1 = up * scaleY;
                c2 = forward * scaleXZ;
            }
            else
            {
                c0 = new float3(cosY * scaleXZ, 0, -sinY * scaleXZ);
                c1 = new float3(0, scaleY, 0);
                c2 = new float3(sinY * scaleXZ, 0, cosY * scaleXZ);
            }

            return new PackedMatrix(c0, c1, c2, worldPosition);
        }
    }
}
