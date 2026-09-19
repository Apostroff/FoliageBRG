using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    public class PrototypesManager : IDisposable
    {
        private const float kMinRelativeHeight = 1f / 2160f; // 1 pixel in 4k resolution
        private readonly BatchRendererGroup _batchRendererGroup;
        private readonly float _cameraFov;
        private NativeHashMap<int, PrototypePrefab> _prototypes;
        
        public PrototypesManager(BatchRendererGroup batchRendererGroup, float cameraFov)
        {
            _batchRendererGroup = batchRendererGroup;
            _prototypes = new NativeHashMap<int, PrototypePrefab>(256, Allocator.Persistent);

            _cameraFov = cameraFov;
        }

        public ref NativeHashMap<int, PrototypePrefab> Prototypes => ref _prototypes;


        public void Dispose()
        {
            foreach (var prototype in _prototypes)
            {
                if (!prototype.Value.LODs.IsCreated)
                    continue;

                foreach (var prototypeLoD in prototype.Value.LODs)
                {
                    if (!prototypeLoD.Meshes.IsCreated)
                        continue;

                    UnregisterMeshAndMaterials(prototypeLoD);
                }

                prototype.Value.Dispose();
            }
            _prototypes.Clear();
            _prototypes.Dispose();
        }

        private void UnregisterMeshAndMaterials(PrototypeLOD prototypeLoD)
        {
            foreach (var meshLOD in prototypeLoD.Meshes)
            {
                if (meshLOD.MeshID != BatchMeshID.Null)
                {
                    _batchRendererGroup?.UnregisterMesh(meshLOD.MeshID);
                    
                }

                if (meshLOD.MaterialID != BatchMaterialID.Null)
                {
                    _batchRendererGroup?.UnregisterMaterial(meshLOD.MaterialID);
                }
            }
        }

        public void RegisterPrototype(int rootId, GameObject root, ref PrototypeSettings settings, 
            out PackedMatrix matrix, out float2 scale, bool reloadPrototype = false)
        {
            int referenceCount = 0;
            if (_prototypes.TryGetValue(rootId, out PrototypePrefab prototypePrefab))
            {
                if (reloadPrototype)
                {
                    referenceCount = prototypePrefab.ReferencesCount;
                    prototypePrefab.Dispose();
                }
                else
                {
                    prototypePrefab.ReferencesCount++;
                    prototypePrefab.Settings = settings;
                    _prototypes[rootId] = prototypePrefab;
                    matrix = prototypePrefab.Matrix;
                    scale = prototypePrefab.Scale;
                    return;
                }
                    
            }

            PrototypePrefab prototypePrefabNew = new PrototypePrefab();
            prototypePrefabNew.ReferencesCount += referenceCount;
            prototypePrefabNew.Settings = settings;
            if (CreateNewPrototype(root, ref prototypePrefabNew))
            {
                if (referenceCount == 0)
                {
                    _prototypes.Add(rootId, prototypePrefabNew);
                }
                else
                {
                    _prototypes[rootId] = prototypePrefabNew;
                }
            }

            matrix = prototypePrefabNew.Matrix;
            scale = prototypePrefabNew.Scale;
        }

        private Renderer FirstRenderer(Renderer[] renderers)
        {
            foreach (var renderer in renderers)
            {
                if (renderer)
                    return renderer;
            }

            return null;
        }
        
        private bool CreateNewPrototype(GameObject root, ref PrototypePrefab prototypePrefab)
        {
            
            prototypePrefab.ReferencesCount = 1;
            prototypePrefab.Matrix = PackedMatrix.identityMatrix;
            prototypePrefab.LoDMeshCount = 0;
            
            if (root.TryGetComponent(out LODGroup lodGroup) && lodGroup.lodCount > 0)
            {
                LOD[] lods = lodGroup.GetLODs();
                prototypePrefab.LODs = new NativeArray<PrototypeLOD>(lods.Length, Allocator.Persistent);
                prototypePrefab.Distances = new NativeArray<float>(lods.Length, Allocator.Persistent);
                prototypePrefab.ScreenRelativeTransitionHeights = new NativeArray<float>(lods.Length, Allocator.Persistent);
                
                if (lods[0].renderers.Length > 0)
                {
                    Renderer renderer = FirstRenderer(lods[0].renderers);
                    if (renderer == null)
                    {
                        Debug.LogWarning("No renderers found for LOD0 " + root.name);
                        return false;
                    }
                    prototypePrefab.Matrix = new PackedMatrix(renderer.transform.localToWorldMatrix);
                    float3 scale = renderer.transform.lossyScale;
                    prototypePrefab.Scale = scale.xy;
                }
                
                Bounds bounds = new Bounds();
                for (int i = 0; i < lodGroup.lodCount; i++)
                {
                    PrototypeLOD lod = new PrototypeLOD();

                    AddLOD(lods[i], ref lod, ref bounds);
                    prototypePrefab.LODs[i] = lod;
                    prototypePrefab.LoDMeshCount += lod.Meshes.Length;
                }

                float widthScale = prototypePrefab.Settings.WidthScale;
                float heightScale = prototypePrefab.Settings.HeightScale;
                Vector3 size = bounds.size;
                bounds.size = new Vector3(size.x * widthScale, size.y * heightScale, size.z * widthScale);
                prototypePrefab.AABBLocal = new AABB
                {
                    Center = bounds.center,
                    Extents = bounds.extents
                };

                for (int i = 0; i < lodGroup.lodCount; i++)
                {
                    prototypePrefab.ScreenRelativeTransitionHeights[i] = lods[i].screenRelativeTransitionHeight;
                    prototypePrefab.Distances[i] = RelativeHeightToDistance(lods[i].screenRelativeTransitionHeight,
                        lodGroup.size, _cameraFov, 1);
                }
            }
            else if (root.TryGetComponent(out Renderer renderer))
            {
                prototypePrefab.LODs = new NativeArray<PrototypeLOD>(1, Allocator.Persistent);
                PrototypeLOD lod = new PrototypeLOD();
                AddRenderer(renderer, ref lod);
                prototypePrefab.LODs[0] = lod;
                prototypePrefab.LoDMeshCount = lod.Meshes.Length;
                prototypePrefab.AABBLocal = new AABB
                {
                    Center = renderer.bounds.center,
                    Extents = renderer.bounds.extents
                };
                prototypePrefab.Distances = new NativeArray<float>(1, Allocator.Persistent);
                prototypePrefab.Distances[0] = RelativeHeightToDistance(kMinRelativeHeight, prototypePrefab.AABBLocal.Size.y, _cameraFov);
                prototypePrefab.ScreenRelativeTransitionHeights = new NativeArray<float>(1, Allocator.Persistent);
                prototypePrefab.ScreenRelativeTransitionHeights[0] = kMinRelativeHeight;
                prototypePrefab.Matrix = new PackedMatrix(renderer.transform.localToWorldMatrix);
                float3 scale = renderer.transform.lossyScale;
                prototypePrefab.Scale = scale.xy;
            }
            else
            {
                return false;
            }

            prototypePrefab.Settings.MaxDistance = math.min(prototypePrefab.Distances[^1], prototypePrefab.Settings.MaxDistance);
            return true;
        }

        private void AddRenderer(Renderer renderer, ref PrototypeLOD prototypePrefabLod)
        {
            int count = renderer.sharedMaterials.Length;
            if (count == 0)
            {
                return;
            }
            
            if (!renderer.TryGetComponent(out MeshFilter meshFilter))
            {
                return;
            }
            
            prototypePrefabLod.Meshes = new NativeArray<MeshLOD>(count, Allocator.Persistent);

            for (int i = 0; i < count; i++)
            {
                var meshLOD = new MeshLOD
                {
                    MeshID = RegisterMesh(meshFilter.sharedMesh),
                    MaterialID = RegisterMaterial(renderer.sharedMaterials[i]),
                    SubmeshIndex = i
                };
                prototypePrefabLod.Meshes[i] = meshLOD;
            }
        }

        private void AddLOD(LOD lod, ref PrototypeLOD prototypePrefabLod, ref Bounds bounds)
        {
            int meshCount = 0;
            foreach (var lodRenderer in lod.renderers)
            {
                if (!lodRenderer)
                    continue;
                Material[] materials = lodRenderer.sharedMaterials;
                if (materials == null || materials.Length == 0)
                    continue;
                
                meshCount += lodRenderer.sharedMaterials.Length;
            }
            prototypePrefabLod.Meshes = new NativeArray<MeshLOD>(meshCount, Allocator.Persistent);
            if (meshCount == 0)
                return;
            
            
            int current = 0;
            foreach (var lodRenderer in lod.renderers)
            {
                bounds.Encapsulate(lodRenderer.bounds);
                if (!lodRenderer.TryGetComponent(out MeshFilter meshFilter))
                {
                    continue;
                }

                for (int j = 0; j < lodRenderer.sharedMaterials.Length; j++)
                {
                    var meshLOD = new MeshLOD
                    {
                        MeshID = RegisterMesh(meshFilter.sharedMesh),
                        MaterialID = RegisterMaterial(lodRenderer.sharedMaterials[j]),
                        SubmeshIndex = j
                    };
                    prototypePrefabLod.Meshes[current++] = meshLOD;
                }
            }
        }

        private BatchMeshID RegisterMesh(Mesh mesh)
        {
            var batchMeshID = _batchRendererGroup.RegisterMesh(mesh);
            return batchMeshID;
        }

        private BatchMaterialID RegisterMaterial(Material material)
        {
            return _batchRendererGroup.RegisterMaterial(material);
        }

        private float RelativeHeightToDistance(
            float relativeHeight, float size, float fieldOfView, float bias = 1)
        {
            if (relativeHeight <= float.Epsilon)
                return float.MaxValue;

            float halfAngle = Mathf.Tan(Mathf.Deg2Rad * fieldOfView * 0.5f);
            float distance = (size * 0.5f) / relativeHeight / halfAngle;
            return distance * bias;
        }

        public void UnregisterPrototype(int prototypeId)
        {
            if (!_prototypes.TryGetValue(prototypeId, out var prototype))
            {
                return;
            }
            prototype.ReferencesCount--;
            if (prototype.ReferencesCount <= 0)
            {
                RemovePrototype(prototypeId);
            }
            else
            {
                _prototypes[prototypeId] = prototype;
            }
        }

        private void RemovePrototype(int prototypeId)
        {
            if (!_prototypes.TryGetValue(prototypeId, out PrototypePrefab prototype)) 
                return;
            
            if (prototype.LODs.IsCreated)
            {
                foreach (var prototypeLoD in prototype.LODs)
                {
                    if (!prototypeLoD.Meshes.IsCreated)
                        continue;
                    UnregisterMeshAndMaterials(prototypeLoD);
                }
            }
            
            prototype.Dispose();
            _prototypes.Remove(prototypeId);
        }
    }
}