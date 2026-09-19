
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace FoliageBRG
{
    public class InstancesManager
    {

        private readonly BatchesContainer _batchesContainer;
        private readonly Dictionary<int, LayoutInstancesData> _layoutData = new();
        private readonly CellLayoutManager _layoutManager;
        private readonly PrototypesManager _prototypesManager;

        public InstancesManager(BatchesContainer batchesContainer,
            CellLayoutManager layoutManager,
            PrototypesManager prototypesManager)
        {
            _layoutManager = layoutManager;
            _batchesContainer = batchesContainer;
            _prototypesManager = prototypesManager;
        }

        public int AddInstance(int prototypeId, int instanceId, ref PackedMatrix matrix, float2 scale)
        {
            float2 position = matrix.GetPosition().xz;
            int layoutId = _layoutManager.GetOrCreateLayout(position, BRGConstants.TreeLayoutSize, BRGConstants.TreePatchCount);
            if (!_layoutData.TryGetValue(layoutId, out LayoutInstancesData batchesByPrototype))
            {
                batchesByPrototype = new LayoutInstancesData();
                _layoutData.Add(layoutId, batchesByPrototype);
            }

            if (!batchesByPrototype.Map.TryGetValue(prototypeId, out var instancesData))
            {
                instancesData = new PrototypeInstanceData();
                batchesByPrototype.Map.Add(prototypeId, instancesData);
            }

            batchesByPrototype.Dirty = true;
            instancesData.Dirty = true;
            instancesData.Map[instanceId] = new ScaleMatrixData(matrix, scale);
            return layoutId;
        }

        public void AddInstancePack(int layoutId, int prototypeId, int packId, PackedMatrix[] matrices)
        {
            if (!_layoutData.TryGetValue(layoutId, out LayoutInstancesData batchesByPrototype))
            {
                batchesByPrototype = new LayoutInstancesData();
                _layoutData.Add(layoutId, batchesByPrototype);
            }
            if (!batchesByPrototype.Map.TryGetValue(prototypeId, out PrototypeInstanceData instancesData))
            {
                instancesData = new PrototypeInstanceData();
                batchesByPrototype.Map.Add(prototypeId, instancesData);
            }

            batchesByPrototype.Dirty = true;
            instancesData.Dirty = true;
            instancesData.Map[packId] = new ScaleMatrixData(matrices);
        }

        public void RemoveInstance(int prototypeId, int layoutId, int instanceId)
        {
            if (!_layoutData.TryGetValue(layoutId, out LayoutInstancesData batchesByLayout))
            {
                return;
            }

            if (!batchesByLayout.Map.TryGetValue(prototypeId, out var instancesData))
            {
                return;
            }

            instancesData.Dirty = instancesData.Map.Remove(instanceId);
            batchesByLayout.Dirty = instancesData.Dirty;
            if (instancesData.Map.Count == 0)
            {
                batchesByLayout.Map.Remove(prototypeId);
            }
        }

        public void RayInterset(Ray ray, List<(int, float)> ids)
        {
            foreach (KeyValuePair<int, LayoutInstancesData> layoutPair in _layoutData)
            {
                CellLayout layout = _layoutManager.CellLayouts[layoutPair.Key];
                Bounds bounds = new Bounds(layout.GlobalBounds.Center, layout.GlobalBounds.Size);
                if (!bounds.IntersectRay(ray))
                {
                    continue;
                }
                foreach (var prototypeData in layoutPair.Value.Map)
                {
                    if (!_prototypesManager.Prototypes.TryGetValue(prototypeData.Key, out var prototype))
                    {
                        continue;
                    }
                    
                    AABB localAABB = prototype.AABBLocal;
                    foreach (var instanceData in prototypeData.Value.Map)
                    {
                        AABB world  = AABB.Transform(instanceData.Value.Matrix.fullMatrix, localAABB);
                        Bounds worldBounds = new Bounds(world.Center, world.Size);
                        if (worldBounds.IntersectRay(ray, out float distance))
                        {
                            ids.Add((instanceData.Key, distance));
                        }
                    }
                }
            }
        }
        
        public bool Update()
        {
            bool dirty = false;
            foreach (KeyValuePair<int, LayoutInstancesData> layoutInstancesData in _layoutData)
            {
                if (layoutInstancesData.Value.Dirty)
                {
                    UpdateLayout(layoutInstancesData.Key, layoutInstancesData.Value);
                    layoutInstancesData.Value.Dirty = false;
                    dirty = true;
                }
            }

            return dirty;
        }

        private void UpdatePrototype(int layoutId, int prototypeId, PrototypeInstanceData prototypeData,
            ref CellLayout layout, ref NativeList<InstanceBatch> batches)
        {

            if (!_prototypesManager.Prototypes.TryGetValue(prototypeId, out var prototype))
            {
                Debug.Log("Can't find prototype " + prototypeId);
                return;
            }

            InstanceBatch batch = new InstanceBatch(prototypeId, BRGConstants.TreePatchCount, layoutId);
            int cellCount = BRGConstants.TreePatchCount * BRGConstants.TreePatchCount;
            NativeArray<NativeList<PackedMatrix>> matrices =
                new NativeArray<NativeList<PackedMatrix>>(cellCount, Allocator.Temp);

            NativeArray<NativeList<float2>> scales =
                new NativeArray<NativeList<float2>>(cellCount, Allocator.Temp);

            for (int i = 0; i < cellCount; i++)
            {
                matrices[i] = new NativeList<PackedMatrix>(Allocator.Temp);
                scales[i] = new NativeList<float2>(Allocator.Temp);
            }

            foreach (var instances in prototypeData.Map)
            {
                ScaleMatrixData matrixData = instances.Value;

                if (matrixData.Pack == null)
                {
                    ProcessMatrix(layoutId, ref layout, ref matrixData.Matrix, matrixData.Scale, ref prototype, ref matrices, ref scales);
                }
                else
                {
                    for (var i = 0; i < matrixData.Pack.Length; i++)
                    {
                        var matrix = matrixData.Pack[i];
                        ProcessMatrix(layoutId, ref layout, ref matrix, matrixData.Scale, ref prototype, ref matrices,
                            ref scales);
                    }
                }
            }

            int count = 0;
            for (int i = 0; i < cellCount; i++)
            {
                NativeList<PackedMatrix> cellMatrices = matrices[i];
                NativeList<float2> cellScales = scales[i];
                if (cellMatrices.Length <= 0)
                    continue;

                batch.AddMatrixRange(ref cellMatrices,  ref cellScales);
                InstancesChunk chunk = new InstancesChunk(count, (uint) cellMatrices.Length, i);
                batch.AddChunk(ref chunk);
                count +=  cellMatrices.Length;
            }

            GraphicsBuffer buffer = _batchesContainer.RefreshBuffer(ref batch);
            _batchesContainer.AddBuffer(layoutId, buffer);
            batches.Add(batch);
        }

        private void ProcessMatrix(int layoutId, ref CellLayout layout, 
            ref PackedMatrix instanceMatrix, float2 instanceScale, 
            ref PrototypePrefab prototype, 
            ref NativeArray<NativeList<PackedMatrix>> matrices, 
            ref NativeArray<NativeList<float2>> scales)
        {
            PackedMatrix matrix = new PackedMatrix(math.mul(instanceMatrix.fullMatrix, prototype.Matrix.fullMatrix));
            float2 scale = instanceScale * prototype.Scale;
            float2 position = instanceMatrix.GetPosition().xz;
            int cellIndex = layout.CellIndex(position);
            if (cellIndex == -1)
            {
                Debug.LogWarning("Can't find cell for position " + position + " " + layout.GlobalBounds);
                return;
            }

            matrices[cellIndex].Add(matrix);
            scales[cellIndex].Add(scale);
            _layoutManager.AddBounds(layoutId, cellIndex, matrix, prototype);
        }

        private void UpdateLayout(int layoutId, LayoutInstancesData layoutInstances)
        {
            _batchesContainer.DisposeByLayoutId(layoutId);
            NativeList<InstanceBatch> batches = _batchesContainer.GetBatches(layoutId);
            CellLayout cellLayout = _layoutManager.CellLayouts[layoutId];
            foreach (var prototype in layoutInstances.Map)
            {
                UpdatePrototype(layoutId, prototype.Key, prototype.Value, ref cellLayout, ref batches);
                prototype.Value.Dirty = false;
            }
        }
    }

    class LayoutInstancesData
    {
        public bool Dirty;

        public Dictionary<int, PrototypeInstanceData> Map = new();
    }

    class PrototypeInstanceData
    {
        public bool Dirty;
        public Dictionary<int, ScaleMatrixData> Map = new();
    }

    class ScaleMatrixData
    {
        public PackedMatrix Matrix;
        public float2 Scale;
        public PackedMatrix[] Pack;
        public ScaleMatrixData(PackedMatrix matrix, float2 scale)
        {
            Matrix = matrix;
            Scale = scale;
            Pack = null;
        }

        public ScaleMatrixData(PackedMatrix[] pack)
        {
            Pack =  pack;
            Scale = new float2(1, 1);
            Matrix = PackedMatrix.identityMatrix;
        }
    }
}
