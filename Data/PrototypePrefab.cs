using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    public struct PrototypePrefab : IDisposable
    {
        public AABB AABBLocal;
        public NativeArray<float> Distances;
        public int LoDMeshCount;
        public NativeArray<PrototypeLOD> LODs;
        public int ReferencesCount;
        public NativeArray<float> ScreenRelativeTransitionHeights;
        public PrototypeSettings Settings;
        public PackedMatrix Matrix;
        public float2 Scale;

        public void Dispose()
        {
            if (LODs.IsCreated)
            {
                for (int i = 0; i < LODs.Length; i++)
                    LODs[i].Dispose();
            }

            LODs.Dispose();
            Distances.Dispose();
            ScreenRelativeTransitionHeights.Dispose();
        }
    }

    public struct PrototypeLOD : IDisposable
    {
        public NativeArray<MeshLOD> Meshes;

        public void Dispose()
        {
            Meshes.Dispose();
        }
    }

    public struct MeshLOD
    {
        public BatchMeshID MeshID;
        public BatchMaterialID MaterialID;
        public int SubmeshIndex;
    }

    public struct PrototypeSettings
    {
        public float MaxDistance;
        public float MaxShadowDistance;
        public bool PerInstanceCulling;
        public float HeightScale;
        public float WidthScale;
    }
}