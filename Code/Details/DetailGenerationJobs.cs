using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace FoliageBRG.Details
{
    
    // Counts the instances of every (prototype, patch) entry and computes the patch bounds in terrain-local space.
    // Walks exactly the same random stream as <see cref="FillDetailInstancesJob"/>, so the counts match the fill.
    
    [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
    public struct CountDetailInstancesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<DetailPatchEntry> Entries;
        [ReadOnly] public NativeArray<DetailPrototypeParams> Prototypes;
        [ReadOnly] public NativeArray<byte> Coverage;
        [ReadOnly] public NativeArray<float> Heights;
        [ReadOnly] public NativeArray<byte> Holes;
        public DetailTerrainParams Terrain;

        [WriteOnly] public NativeArray<int> Counts;
        [WriteOnly] public NativeArray<AABB> Bounds;

        public void Execute(int index)
        {
            DetailPatchEntry entry = Entries[index];
            DetailPrototypeParams prototype = Prototypes[entry.PrototypeSlot];
            int patchSize = Terrain.PatchSize;
            float cellSizeX = Terrain.CellSizeX;
            float cellSizeZ = Terrain.CellSizeZ;
            int baseX = entry.PatchX * patchSize;
            int baseZ = entry.PatchY * patchSize;

            int accepted = 0;
            float3 min = new float3(float.MaxValue);
            float3 max = new float3(float.MinValue);

            for (int z = 0; z < patchSize; z++)
            {
                int row = entry.CoverageOffset + z * patchSize;
                int cellZ = baseZ + z;
                for (int x = 0; x < patchSize; x++)
                {
                    byte coverage = Coverage[row + x];
                    if (coverage == 0)
                        continue;

                    int cellX = baseX + x;
                    Random rng = DetailPlacement.CellRandom(cellX, cellZ, prototype.PrototypeIndex, prototype.NoiseSeed);
                    int count = DetailPlacement.CellInstanceCount(Terrain, prototype, coverage, ref rng);
                    for (int k = 0; k < count; k++)
                    {
                        float localX = (cellX + rng.NextFloat()) * cellSizeX;
                        float localZ = (cellZ + rng.NextFloat()) * cellSizeZ;
                        DetailPlacement.RandomRotation(ref rng); // keeps the stream identical to the fill job

                        if (DetailPlacement.IsHole(Holes, Terrain.HolesResolution, Terrain.Size, localX, localZ, prototype.HoleTestRadius))
                            continue;

                        float height = DetailPlacement.SampleHeight(Heights, Terrain.HeightmapResolution, Terrain.Size, localX, localZ);
                        float3 position = new float3(localX, height, localZ);
                        min = math.min(min, position);
                        max = math.max(max, position);
                        accepted++;
                    }
                }
            }

            Counts[index] = accepted;

            AABB bounds = default;
            if (accepted > 0)
            {
                // Pad by the prototype bounds. The rotation is not known here, so the larger horizontal extent is used on both axes.
                float3 extents = prototype.LocalBounds.Extents;
                float3 center = prototype.LocalBounds.Center;
                float horizontal = math.max(math.abs(center.x), math.abs(center.z)) + math.max(extents.x, extents.z);
                min += new float3(-horizontal, math.min(0f, center.y - extents.y), -horizontal);
                max += new float3(horizontal, math.max(0f, center.y + extents.y), horizontal);
                bounds.Center = (min + max) * 0.5f;
                bounds.Extents = (max - min) * 0.5f;
            }

            Bounds[index] = bounds;
        }
    }

    /// <summary>
    /// Writes the instance matrices and scales of every entry of one prototype batch at entry.InstanceStart.
    /// Entries must have InstanceStart and InstanceCount filled from the count pass.
    /// </summary>
    [BurstCompile(OptimizeFor = OptimizeFor.Performance)]
    public struct FillDetailInstancesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<DetailPatchEntry> Entries;
        [ReadOnly] public NativeArray<DetailPrototypeParams> Prototypes;
        [ReadOnly] public NativeArray<byte> Coverage;
        [ReadOnly] public NativeArray<float> Heights;
        [ReadOnly] public NativeArray<byte> Holes;
        public DetailTerrainParams Terrain;

        [NativeDisableParallelForRestriction] public NativeArray<PackedMatrix> Transforms;
        [NativeDisableParallelForRestriction] public NativeArray<float2> Scales;

        public void Execute(int index)
        {
            DetailPatchEntry entry = Entries[index];
            int target = entry.InstanceCount;
            if (target == 0)
                return;

            DetailPrototypeParams prototype = Prototypes[entry.PrototypeSlot];
            int patchSize = Terrain.PatchSize;
            float cellSizeX = Terrain.CellSizeX;
            float cellSizeZ = Terrain.CellSizeZ;
            int baseX = entry.PatchX * patchSize;
            int baseZ = entry.PatchY * patchSize;
            float3 defaultUp = new float3(0, 1, 0);

            int written = 0;
            for (int z = 0; z < patchSize && written < target; z++)
            {
                int row = entry.CoverageOffset + z * patchSize;
                int cellZ = baseZ + z;
                for (int x = 0; x < patchSize && written < target; x++)
                {
                    byte coverage = Coverage[row + x];
                    if (coverage == 0)
                        continue;

                    int cellX = baseX + x;
                    Random rng = DetailPlacement.CellRandom(cellX, cellZ, prototype.PrototypeIndex, prototype.NoiseSeed);
                    int count = DetailPlacement.CellInstanceCount(Terrain, prototype, coverage, ref rng);
                    for (int k = 0; k < count && written < target; k++)
                    {
                        float localX = (cellX + rng.NextFloat()) * cellSizeX;
                        float localZ = (cellZ + rng.NextFloat()) * cellSizeZ;
                        float rotationY = DetailPlacement.RandomRotation(ref rng);

                        if (DetailPlacement.IsHole(Holes, Terrain.HolesResolution, Terrain.Size, localX, localZ, prototype.HoleTestRadius))
                            continue;

                        float height = DetailPlacement.SampleHeight(Heights, Terrain.HeightmapResolution, Terrain.Size, localX, localZ);
                        float t = DetailPlacement.ScaleNoise(prototype, localX, localZ);
                        float scaleXZ = math.lerp(prototype.MinWidth, prototype.MaxWidth, t);
                        float scaleY = math.lerp(prototype.MinHeight, prototype.MaxHeight, t);
                        float3 up = prototype.AlignToGround
                            ? DetailPlacement.SampleNormal(Heights, Terrain.HeightmapResolution, Terrain.Size, localX, localZ)
                            : defaultUp;
                        float3 world = new float3(localX, height, localZ) + Terrain.Origin;

                        int slot = entry.InstanceStart + written;
                        Transforms[slot] = DetailPlacement.BuildMatrix(world, rotationY, scaleXZ, scaleY, prototype.AlignToGround, up);
                        Scales[slot] = new float2(scaleXZ, scaleY);
                        written++;
                    }
                }
            }

            // Never expected: the count pass walked the same stream. Keeps the buffer defined if it ever diverges.
            for (; written < target; written++)
            {
                int slot = entry.InstanceStart + written;
                Transforms[slot] = new PackedMatrix(float3.zero, float3.zero, float3.zero, Terrain.Origin);
                Scales[slot] = float2.zero;
            }
        }
    }
}
