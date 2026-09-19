using System;
using Unity.Collections;
using Unity.Mathematics;

namespace FoliageBRG.Details
{
    // Snapshot of one detail prototype (one detail layer of a TerrainData) for the scatter jobs.
    public struct DetailPrototypeParams
    {
        //Index in TerrainData.detailPrototypes, which is also the detail layer index
        public int PrototypeIndex;

        //Instance id of the prototype root, the key used by PrototypesManager
        public int RootId;
        
        // TerrainData.ComputeDetailCoverage: instances per metre along one axis at coverage 255.
        // Unity derives it from density, mean width and the mesh width, so it is taken from the API rather than recomputed.
        public float CoveragePerMeter;

        public float MinWidth;
        public float MaxWidth;
        public float MinHeight;
        public float MaxHeight;

        public bool AlignToGround;

        public float NoiseSpread;
        public int NoiseSeed;

        // Hole test radius in metres (DetailPrototype.holeEdgePadding times the mesh width). 0 disables the radius test
        public float HoleTestRadius;

        // Prototype bounds registered in the BRG (already scaled by max width/height), used to pad the patch bounds
        public AABB LocalBounds;
    }
    // One (prototype, patch) pair with non-zero coverage. Entries are stored prototype-major, patches in cell index order.
    public struct DetailPatchEntry
    {
        public int PrototypeSlot;
        public int PatchX;
        public int PatchY;

        //PatchY * patchCount + PatchX, the cell index used by CellLayout and InstancesChunk
        public int CellIndex;

        //Byte offset of this patch's coverage block (patchSize * patchSize bytes) inside DetailSourceData.Coverage
        public int CoverageOffset;

        //Offset of the first instance of this patch inside its prototype batch. Filled after counting
        public int InstanceStart;

        //Number of instances that survive the hole test. Filled after counting
        public int InstanceCount;
    }

    public struct DetailTerrainParams
    {
        public float3 Size;

        //World position of the terrain, added to the local instance positions
        public float3 Origin;

        public int DetailResolution;
        public int PatchSize;
        public int PatchCount;
        public int HeightmapResolution;

        // 0 when the terrain has no holes data
        public int HolesResolution;

        //DetailScatterMode.InstanceCountMode: the coverage byte is the instance count of the cell
        public bool InstanceCountMode;

        public float CellSizeX => Size.x / DetailResolution;
        public float CellSizeZ => Size.z / DetailResolution;
    }

    //Main-thread time of every phase of one generation, for the timing log
    public struct DetailGenerationStats
    {
        public double ReadMs;
        public double CountMs;
        public double FillMs;
        public double BoundsMs;
    }

    // Everything the scatter jobs read, copied out of a TerrainData into native memory on the main thread.
    public struct DetailSourceData : IDisposable
    {
        public DetailTerrainParams Terrain;
        public NativeArray<DetailPrototypeParams> Prototypes;
        public NativeArray<DetailPatchEntry> Entries;

        //Prototypes.Length + 1 entries: first entry index of every prototype slot, last element = Entries.Length
        public NativeArray<int> EntryStartPerPrototype;

        //Coverage bytes, one block of PatchSize * PatchSize per entry, rows along Z, columns along X
        public NativeArray<byte> Coverage;

        //Normalized heights [0..1], HeightmapResolution squared, row-major [z, x]
        public NativeArray<float> Heights;

        //1 = surface, 0 = hole, HolesResolution squared, row-major [z, x]. Empty when HolesResolution is 0
        public NativeArray<byte> Holes;

        public bool IsCreated => Entries.IsCreated;

        public void Dispose()
        {
            if (Prototypes.IsCreated) 
                Prototypes.Dispose();
            
            if (Entries.IsCreated) 
                Entries.Dispose();
            
            if (EntryStartPerPrototype.IsCreated) 
                EntryStartPerPrototype.Dispose();
            
            if (Coverage.IsCreated) 
                Coverage.Dispose();
            
            if (Heights.IsCreated) 
                Heights.Dispose();
            
            if (Holes.IsCreated) 
                Holes.Dispose();
        }
    }
}
