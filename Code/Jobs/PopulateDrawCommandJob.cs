using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine.Rendering;

namespace FoliageBRG
{
    [BurstCompile]
    public unsafe struct PopulateDrawCommandJob : IJob
    {
        [ReadOnly] public NativeArray<int> Counters;

        [ReadOnly] public NativeArray<DrawCallData> DrawCalls;

        [NativeDisableUnsafePtrRestriction] public BatchCullingOutputDrawCommands* OutputDrawCommands;
 
        
        
        public void Execute()
        {
            int drawCallCount = Counters[0];
            int count = Counters[1];
            if (drawCallCount == 0 || count == 0)
                return;

            int alignment = UnsafeUtility.AlignOf<long>();

            var drawCommands = OutputDrawCommands;
            drawCommands->drawCommands = (BatchDrawCommand*) UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawCommand>() * drawCallCount, alignment, Allocator.TempJob);

            drawCommands->drawRanges = (BatchDrawRange*) UnsafeUtility.Malloc(UnsafeUtility.SizeOf<BatchDrawRange>(), alignment, Allocator.TempJob);
            drawCommands->visibleInstances = (int*) UnsafeUtility.Malloc(count * sizeof(int), alignment, Allocator.TempJob);
            drawCommands->drawCommandPickingInstanceIDs = null;

            drawCommands->drawCommandCount = drawCallCount;
            drawCommands->drawRangeCount = 1;
            drawCommands->visibleInstanceCount = count;


            drawCommands->instanceSortingPositions = null;
            drawCommands->instanceSortingPositionFloatCount = 0;

            uint indexOffset = 0;
            uint visibleOffset = 0;
            for (int i = 0; i < DrawCalls.Length; i++)
            {
                DrawCallData drawCall = DrawCalls[i];

                if (drawCall.Indices != null) // Batch Start
                {
                    UnsafeUtility.MemCpy(drawCommands->visibleInstances + indexOffset, drawCall.Indices, drawCall.InstanceCount * sizeof(int));
                    UnsafeUtility.FreeTracked(drawCall.Indices, Allocator.TempJob);
                    visibleOffset = indexOffset;
                    indexOffset += drawCall.InstanceCount;
                }

                drawCommands->drawCommands[i].visibleOffset = visibleOffset;
                drawCommands->drawCommands[i].visibleCount = drawCall.InstanceCount;
                drawCommands->drawCommands[i].batchID = drawCall.BatchID;
                drawCommands->drawCommands[i].materialID = drawCall.MaterialID;
                drawCommands->drawCommands[i].meshID = drawCall.MeshID;
                drawCommands->drawCommands[i].submeshIndex = (ushort) drawCall.SubmeshIndex;
                drawCommands->drawCommands[i].splitVisibilityMask = 0xff;
                drawCommands->drawCommands[i].flags = 0;
                drawCommands->drawCommands[i].sortingPosition = 0;


            }

            // Configure the single draw range to cover the single draw command which
            // is at offset 0.
            drawCommands->drawRanges[0].drawCommandsType = BatchDrawCommandType.Direct;
            drawCommands->drawRanges[0].drawCommandsBegin = 0;
            drawCommands->drawRanges[0].drawCommandsCount = (uint) drawCallCount;

            BatchFilterSettings filterSettings = new BatchFilterSettings
            {
                renderingLayerMask = 0xffffffff,
                shadowCastingMode = ShadowCastingMode.On,
                receiveShadows = true,

            };
            drawCommands->drawRanges[0].filterSettings = filterSettings;

        }
    }
}