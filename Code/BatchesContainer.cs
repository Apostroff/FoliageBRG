
using System;
using System.Collections.Generic;
using FoliageBRG.Details;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    public class BatchesContainer : IDisposable
    {
        private const int kSizeOfMatrix = sizeof(float) * 4 * 4;
        private const int kSizeOfPackedMatrix = sizeof(float) * 4 * 3;
        private const int kExtraBytes = kSizeOfPackedMatrix * 2;
        private const int kSizeOfFloat2 = sizeof(float) * 2;
        private const int kBytesPerInstance = kSizeOfPackedMatrix +  kSizeOfFloat2;
        private static int ObjectToWorldPropId = Shader.PropertyToID("unity_ObjectToWorld");
        private static int ScalePropId = Shader.PropertyToID("_ObjectScale");

        private readonly BatchRendererGroup _batchRendererGroup;
        private readonly Dictionary<int, List<GraphicsBuffer>> _buffersByLayoutId = new();
        private readonly CellLayoutManager _cellLayoutManager;

        public NativeHashMap<int, NativeList<InstanceBatch>> BatchesByLayoutId;

        public BatchesContainer(BatchRendererGroup batchRendererGroup, CellLayoutManager cellLayoutManager)
        {
            _batchRendererGroup = batchRendererGroup;
            _cellLayoutManager = cellLayoutManager;
            BatchesByLayoutId = new NativeHashMap<int, NativeList<InstanceBatch>>(32, Allocator.Persistent);
        }

        public void Dispose()
        {
            foreach (var pair in BatchesByLayoutId)
            {
                if (!pair.Value.IsCreated)
                    continue;

                foreach (var instanceBatch in pair.Value)
                {
                    if (instanceBatch.BatchID != BatchID.Null)
                        _batchRendererGroup.RemoveBatch(instanceBatch.BatchID);

                    instanceBatch.Dispose();
                }

                pair.Value.Dispose();
            }

            BatchesByLayoutId.Dispose();
            foreach (var pair in _buffersByLayoutId)
            {
                if (pair.Value != null)
                {
                    foreach (GraphicsBuffer buffer in pair.Value)
                    {
                        buffer?.Dispose();
                    }
                }
            }

            _buffersByLayoutId.Clear();
        }

        public bool HasBatches(int layoutId)
        {
            if (BatchesByLayoutId.TryGetValue(layoutId, out NativeList<InstanceBatch> batches))
            {
                return batches.IsCreated && batches.Length > 0;
            }

            return false;
        }

        public NativeList<InstanceBatch> GetBatches(int layoutId)
        {
            if (!BatchesByLayoutId.TryGetValue(layoutId, out NativeList<InstanceBatch> batches))
            {
                batches = new NativeList<InstanceBatch>(64, Allocator.Persistent);
                BatchesByLayoutId.Add(layoutId, batches);
            }

            return batches;
        }

        private List<GraphicsBuffer> GetBuffers(int layoutId)
        {
            if (!_buffersByLayoutId.TryGetValue(layoutId, out List<GraphicsBuffer> buffers))
            {
                buffers = new List<GraphicsBuffer>();
                _buffersByLayoutId.Add(layoutId, buffers);
            }

            return buffers;
        }

        public void RefreshAllBuffers(int layoutId)
        {
            List<GraphicsBuffer> buffers = GetBuffers(layoutId);
            foreach (GraphicsBuffer graphicsBuffer in buffers)
            {
                graphicsBuffer?.Dispose();
            }

            buffers.Clear();

            NativeList<InstanceBatch> batches = GetBatches(layoutId);
            if (!batches.IsCreated)
                return;

            int count = batches.Length;
            for (int i = 0; i < count; i++)
            {
                ref InstanceBatch batch = ref batches.ElementAt(i);
                GraphicsBuffer buffer = RefreshBuffer(ref batch);
                buffers.Add(buffer);
            }
        }

        public GraphicsBuffer RefreshBuffer(ref InstanceBatch instanceBatch, GraphicsBuffer buffer = null)
        {
            if (instanceBatch.BatchID != BatchID.Null)
            {
                _batchRendererGroup.RemoveBatch(instanceBatch.BatchID);
            }

            if (buffer != null && buffer.IsValid())
            {
                buffer.Release();
            }

            if (instanceBatch.InstanceCount == 0)
                return null;

            GraphicsBuffer result = new GraphicsBuffer(GraphicsBuffer.Target.Raw,
                BufferCountForInstances(kBytesPerInstance, instanceBatch.InstanceCount,
                    kExtraBytes), sizeof(int));

            instanceBatch.BatchID = PopulateInstanceDataBuffer(ref instanceBatch, result);
            return result;
        }

        public void AddBuffer(int layoutId, GraphicsBuffer buffer)
        {
            GetBuffers(layoutId).Add(buffer);
        }


        private BatchID PopulateInstanceDataBuffer(ref InstanceBatch instanceBatch, GraphicsBuffer buffer)
        {
            Matrix4x4[] zero = {Matrix4x4.zero};
            int byteAddressObjectToWorld = kSizeOfPackedMatrix * 2;
            buffer.SetData(zero, 0, 0, 1);
            int byteAddressScales = byteAddressObjectToWorld + kSizeOfPackedMatrix * instanceBatch.Transforms.Length;
            
            buffer.SetData(instanceBatch.Transforms.AsArray(), 0,
                byteAddressObjectToWorld / kSizeOfPackedMatrix,
                instanceBatch.Transforms.Length);
            
            buffer.SetData(instanceBatch.Scales.AsArray(), 0, byteAddressScales / kSizeOfFloat2,
                instanceBatch.Scales.Length);

            var metadata = new NativeArray<MetadataValue>(2, Allocator.Temp);
            metadata[0] = new MetadataValue
            {
                NameID = ObjectToWorldPropId,
                Value = (uint)(0x80000000 | byteAddressObjectToWorld),
            };
            metadata[1] = new MetadataValue
            {
                NameID = ScalePropId,
                Value = (uint)(0x80000000 | byteAddressScales),
            };

            BatchID batchID = _batchRendererGroup.AddBatch(metadata, buffer.bufferHandle);
            return batchID;
        }

        int BufferCountForInstances(int bytesPerInstance, int numInstances, int extraBytes = 0)
        {
            // Round byte counts to int multiples
            bytesPerInstance = (bytesPerInstance + sizeof(int) - 1) / sizeof(int) * sizeof(int);
            extraBytes = (extraBytes + sizeof(int) - 1) / sizeof(int) * sizeof(int);
            int totalBytes = bytesPerInstance * numInstances + extraBytes;
            return totalBytes / sizeof(int);
        }


        public int LoadAllDetails(TerrainData terrainData, int layoutId, Vector3 origin)
        {
            NativeList<InstanceBatch> batches = GetBatches(layoutId);
            int maxInstances = TerrainDetailGenerator.Generate(terrainData, layoutId, origin, _cellLayoutManager, batches,
                out DetailGenerationStats stats);

            var uploadStopwatch = System.Diagnostics.Stopwatch.StartNew();
            RefreshAllBuffers(layoutId);
            double uploadMs = uploadStopwatch.Elapsed.TotalMilliseconds;

            if (TerrainDetailGenerator.LogTimings)
                TerrainDetailGenerator.LogTimingsLine(terrainData.name, stats, uploadMs, 1);

            return maxInstances;
        }

        // Moves batches that were uploaded outside the container (DetailBuildScheduler) into the layout, making them visible to culling
        public void AppendPreparedBatches(int layoutId, NativeList<InstanceBatch> prepared, List<GraphicsBuffer> buffers)
        {
            GetBatches(layoutId).AddRange(prepared.AsArray());
            GetBuffers(layoutId).AddRange(buffers);
        }

        // Releases a batch that was uploaded but never appended to a layout: BRG batch, GPU buffer and native lists
        public void ReleaseUnswappedBatch(ref InstanceBatch batch, GraphicsBuffer buffer)
        {
            if (batch.BatchID != BatchID.Null)
            {
                _batchRendererGroup.RemoveBatch(batch.BatchID);
                batch.BatchID = BatchID.Null;
            }

            buffer?.Dispose();
            batch.Dispose();
        }

        
        public void DisposeByLayoutId(int layoutId)
        {
            if (!BatchesByLayoutId.IsCreated)
                return;
            
            if (BatchesByLayoutId.TryGetValue(layoutId, out NativeList<InstanceBatch> batches))
            {
                if (batches.IsCreated)
                {
                    foreach (var batch in batches)
                    {
                        if (batch.BatchID != BatchID.Null)
                            _batchRendererGroup.RemoveBatch(batch.BatchID);

                        batch.Dispose();
                    }
                    batches.Dispose();
                }

                BatchesByLayoutId.Remove(layoutId);
            }

            if (_buffersByLayoutId.TryGetValue(layoutId, out var buffers))
            {
                if (buffers != null)
                {
                    foreach (GraphicsBuffer buffer in buffers)
                    {
                        buffer?.Dispose();
                    }
                }

                _buffersByLayoutId.Remove(layoutId);
            }
        }
    }
}