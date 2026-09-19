using UnityEngine.Rendering;

namespace FoliageBRG
{
    public struct DrawCallData
    {
        public readonly BatchID BatchID;
        public readonly BatchMaterialID MaterialID;
        public readonly BatchMeshID MeshID;
        public readonly uint InstanceCount;
        public readonly int SubmeshIndex;

        public unsafe int* Indices;

        public DrawCallData(BatchID batchID, BatchMaterialID materialID, BatchMeshID meshID, uint instanceCount, int submeshIndex) : this()
        {
            BatchID = batchID;
            MaterialID = materialID;
            MeshID = meshID;
            InstanceCount = instanceCount;
            SubmeshIndex = submeshIndex;
        }

    }
}
