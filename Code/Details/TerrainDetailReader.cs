using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace FoliageBRG.Details
{
    
    // Main-thread copy of everything the scatter jobs need out of a TerrainData, split into steps so a build can be
    // spread over frames (DetailBuildScheduler) or done at once (<see cref="Read"/>).
    // Coverage is read per patch and only for the layers present in that patch (TerrainData stores it sparsely,
    // GetSupportedLayers exposes that), which is about 6x less data than reading whole layers.
    // The only managed allocations left are the arrays the Unity API returns (GetSupportedLayers, GetDetailLayer,
    // GetHeights, GetHoles); none of them has an overload with a caller provided buffer.
    
    [BurstCompile]
    public static unsafe class TerrainDetailReader
    {
        public static NativeArray<DetailPrototypeParams> BuildPrototypeParams(TerrainData terrainData, Allocator allocator)
        {
            DetailPrototype[] prototypes = terrainData.detailPrototypes;
            bool coverageMode = terrainData.detailScatterMode == DetailScatterMode.CoverageMode;
            FoliageBRGSystem system = FoliageBRGSystem.Instance;

            var list = new NativeList<DetailPrototypeParams>(prototypes.Length, Allocator.Temp);
            for (int i = 0; i < prototypes.Length; i++)
            {
                DetailPrototype prototype = prototypes[i];
                if (!prototype.usePrototypeMesh || prototype.prototype == null)
                    continue;

                GameObject root = BRGUtils.GetRoot(prototype.prototype);
                int rootId = root.GetInstanceID();
                if (system == null || !system.GetPrototype(rootId, out PrototypePrefab prefab))
                    continue;

                list.Add(new DetailPrototypeParams
                {
                    PrototypeIndex = i,
                    RootId = rootId,
                    CoveragePerMeter = coverageMode ? terrainData.ComputeDetailCoverage(i) : 0f,
                    MinWidth = prototype.minWidth,
                    MaxWidth = prototype.maxWidth,
                    MinHeight = prototype.minHeight,
                    MaxHeight = prototype.maxHeight,
                    AlignToGround = prototype.alignToGround >= 0.1f,
                    NoiseSpread = prototype.noiseSpread,
                    NoiseSeed = prototype.noiseSeed,
                    HoleTestRadius = prototype.holeEdgePadding > 0f ? prototype.holeEdgePadding * MeshWidth(root) : 0f,
                    LocalBounds = PaddingBounds(root, prototype, prefab.AABBLocal),
                });
            }

            NativeArray<DetailPrototypeParams> result = list.ToArray(allocator);
            list.Dispose();
            return result;
        }

        // Source with the terrain parameters only; takes ownership of <paramref name="prototypes"/>.
        // Fill it with the Read* steps
        public static DetailSourceData Create(TerrainData terrainData, Vector3 origin, NativeArray<DetailPrototypeParams> prototypes)
        {
            return new DetailSourceData
            {
                Prototypes = prototypes,
                Terrain = new DetailTerrainParams
                {
                    Size = terrainData.size,
                    Origin = origin,
                    DetailResolution = terrainData.detailResolution,
                    PatchSize = terrainData.detailResolutionPerPatch,
                    PatchCount = terrainData.detailPatchCount,
                    HeightmapResolution = terrainData.heightmapResolution,
                    HolesResolution = terrainData.holesResolution,
                    InstanceCountMode = terrainData.detailScatterMode == DetailScatterMode.InstanceCountMode,
                },
            };
        }

        // All steps at once. Dispose the result to release everything
        public static DetailSourceData Read(TerrainData terrainData, Vector3 origin, NativeArray<DetailPrototypeParams> prototypes, Allocator allocator)
        {
            DetailSourceData data = Create(terrainData, origin, prototypes);
            try
            {
                ReadCoverageLayout(terrainData, ref data, allocator);
                for (int e = 0; e < data.Entries.Length; e++)
                    ReadCoverageBlock(terrainData, ref data, e);
                ReadHeights(terrainData, ref data, allocator);
                ReadHoles(terrainData, ref data, allocator);
            }
            catch
            {
                data.Dispose();
                throw;
            }

            return data;
        }

        // Step 1: which (prototype, patch) blocks exist (one GetSupportedLayers per patch), the entry table in
        // prototype-major order and the coverage storage. The coverage bytes are filled by <see cref="ReadCoverageBlock"/>.
        
        public static void ReadCoverageLayout(TerrainData terrainData, ref DetailSourceData data, Allocator allocator)
        {
            int patchCount = data.Terrain.PatchCount;
            int patchSize = data.Terrain.PatchSize;
            int cellsPerPatch = patchSize * patchSize;
            int patches = patchCount * patchCount;
            int slots = data.Prototypes.Length;

            // Layers are detail prototype indices; slots are the registered subset (prototypes without a mesh or
            // not registered in the BRG are skipped by TerrainGrassRenderer), so slot != layer in general.
            int maxLayer = 0;
            for (int s = 0; s < slots; s++)
                maxLayer = math.max(maxLayer, data.Prototypes[s].PrototypeIndex);
            int words = maxLayer / 64 + 1;

            var registered = new NativeArray<ulong>(words, Allocator.Temp);
            for (int s = 0; s < slots; s++)
                SetBit(registered, 0, words, data.Prototypes[s].PrototypeIndex);

            var present = new NativeArray<ulong>(patches * words, Allocator.Temp);
            int totalEntries = 0;
            for (int py = 0; py < patchCount; py++)
            for (int px = 0; px < patchCount; px++)
            {
                int cellIndex = py * patchCount + px;
                int[] layers = terrainData.GetSupportedLayers(px * patchSize, py * patchSize, patchSize, patchSize);
                foreach (int layer in layers)
                {
                    if (layer > maxLayer || !HasBit(registered, 0, words, layer))
                        continue;

                    SetBit(present, cellIndex, words, layer);
                    totalEntries++;
                }
            }

            data.Entries = new NativeArray<DetailPatchEntry>(totalEntries, allocator, NativeArrayOptions.UninitializedMemory);
            data.EntryStartPerPrototype = new NativeArray<int>(slots + 1, allocator, NativeArrayOptions.UninitializedMemory);
            data.Coverage = new NativeArray<byte>(totalEntries * cellsPerPatch, allocator, NativeArrayOptions.UninitializedMemory);

            int entry = 0;
            for (int s = 0; s < slots; s++)
            {
                data.EntryStartPerPrototype[s] = entry;
                int layer = data.Prototypes[s].PrototypeIndex;
                for (int py = 0; py < patchCount; py++)
                for (int px = 0; px < patchCount; px++)
                {
                    int cellIndex = py * patchCount + px;
                    if (!HasBit(present, cellIndex, words, layer))
                        continue;

                    data.Entries[entry] = new DetailPatchEntry
                    {
                        PrototypeSlot = s,
                        PatchX = px,
                        PatchY = py,
                        CellIndex = cellIndex,
                        CoverageOffset = entry * cellsPerPatch,
                    };
                    entry++;
                }
            }

            data.EntryStartPerPrototype[slots] = entry;
            present.Dispose();
            registered.Dispose();
        }

        // Step 2, one entry at a time: GetDetailLayer of that patch and layer into the entry's coverage block.</summary>
        public static void ReadCoverageBlock(TerrainData terrainData, ref DetailSourceData data, int entryIndex)
        {
            DetailPatchEntry entry = data.Entries[entryIndex];
            int patchSize = data.Terrain.PatchSize;
            int layer = data.Prototypes[entry.PrototypeSlot].PrototypeIndex;
            int[,] map = terrainData.GetDetailLayer(entry.PatchX * patchSize, entry.PatchY * patchSize, patchSize, patchSize, layer);
            byte* coverage = (byte*) data.Coverage.GetUnsafePtr() + entry.CoverageOffset;
            fixed (int* source = map)
            {
                ConvertCoverage(source, coverage, patchSize * patchSize);
            }
        }

        public static void ReadHeights(TerrainData terrainData, ref DetailSourceData data, Allocator allocator)
        {
            int resolution = data.Terrain.HeightmapResolution;
            float[,] heights = terrainData.GetHeights(0, 0, resolution, resolution); // normalized [0..1], [z, x]
            data.Heights = new NativeArray<float>(resolution * resolution, allocator, NativeArrayOptions.UninitializedMemory);
            fixed (float* source = heights)
            {
                UnsafeUtility.MemCpy(data.Heights.GetUnsafePtr(), source, (long) resolution * resolution * sizeof(float));
            }
        }

        public static void ReadHoles(TerrainData terrainData, ref DetailSourceData data, Allocator allocator)
        {
            int resolution = data.Terrain.HolesResolution;
            if (resolution <= 0)
            {
                data.Holes = new NativeArray<byte>(0, allocator);
                return;
            }

            bool[,] holes = terrainData.GetHoles(0, 0, resolution, resolution); // true = surface, [z, x]
            data.Holes = new NativeArray<byte>(resolution * resolution, allocator, NativeArrayOptions.UninitializedMemory);
            fixed (bool* source = holes)
            {
                UnsafeUtility.MemCpy(data.Holes.GetUnsafePtr(), source, (long) resolution * resolution);
            }
        }

        private static void SetBit(NativeArray<ulong> bits, int row, int words, int bit)
        {
            int index = row * words + (bit >> 6);
            bits[index] = bits[index] | (1ul << (bit & 63));
        }

        private static bool HasBit(NativeArray<ulong> bits, int row, int words, int bit)
        {
            return (bits[row * words + (bit >> 6)] & (1ul << (bit & 63))) != 0;
        }

        private static float MeshWidth(GameObject root)
        {
            MeshFilter filter = root.GetComponentInChildren<MeshFilter>(true);
            return filter != null && filter.sharedMesh != null ? filter.sharedMesh.bounds.size.x : 1f;
        }

        // Bounds used to pad the patch AABBs: the prototype's own renderer bounds scaled by this terrain's max width/height,
        // unioned with what the BRG registered (that one may carry another terrain's scale, or none for Renderer-only prefabs).
        private static AABB PaddingBounds(GameObject root, DetailPrototype prototype, AABB registered)
        {
            Renderer renderer = root.GetComponentInChildren<Renderer>(true);
            if (renderer == null)
                return registered;

            Bounds bounds = renderer.bounds;
            float3 scale = new float3(prototype.maxWidth, prototype.maxHeight, prototype.maxWidth);
            float3 ownMin = (float3) bounds.center * scale - (float3) bounds.extents * scale;
            float3 ownMax = (float3) bounds.center * scale + (float3) bounds.extents * scale;
            float3 min = math.min(ownMin, registered.Min);
            float3 max = math.max(ownMax, registered.Max);
            return new AABB { Center = (min + max) * 0.5f, Extents = (max - min) * 0.5f };
        }

        [BurstCompile]
        private static void ConvertCoverage(int* source, byte* destination, int count)
        {
            for (int i = 0; i < count; i++)
                destination[i] = (byte) math.min(source[i], 255);
        }
    }
}
