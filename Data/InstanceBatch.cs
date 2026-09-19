using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    // one proto one terrain
    public struct InstanceBatch : IDisposable
    {
        public int Layout;
        public int InstanceCount;

        public NativeList<PackedMatrix> Transforms;
        public NativeList<float2> Scales; // Need separate scale for billboards
        public NativeList<InstancesChunk> InstancesChunks;
        public BatchID BatchID;

        private readonly int _terrainPatchCount;
        private readonly int _prototypeId;

        public int PrototypeId => _prototypeId;

        public InstanceBatch(int prototypeId, int terrainPatchCount, int layoutId)
        {
            _prototypeId = prototypeId;
            _terrainPatchCount = terrainPatchCount;
            Transforms = new NativeList<PackedMatrix>(128, Allocator.Persistent);
            Scales = new NativeList<float2>(128, Allocator.Persistent);
            Layout = layoutId;
            InstancesChunks = new NativeList<InstancesChunk>(128, Allocator.Persistent);
            BatchID = BatchID.Null;
            InstanceCount = 0;
        }

        public void AddMatrixRange(ref NativeList<PackedMatrix> matrices, ref NativeList<float2> scales)
        {
            Transforms.AddRange(matrices.AsArray());
            Scales.AddRange(scales.AsArray());
        }

        public void AddChunk(ref InstancesChunk chunk)
        {
            InstancesChunks.Add(chunk);
            InstanceCount += (int) chunk.IndicesCount;
        }

        public void Dispose()
        {
            Transforms.Dispose();
            InstancesChunks.Dispose();
            Scales.Dispose();
        }
    }
}
