using System;
using Unity.Collections;
using Unity.Mathematics;

namespace FoliageBRG
{
    public class CellLayoutManager : IDisposable
    {
        private const int kMaxLayouts = 400 * 2; // tree + grass 
        private NativeArray<CellLayout> _cellLayouts = new(kMaxLayouts, Allocator.Persistent);
        private NativeList<int> _freeIndices = new(kMaxLayouts, Allocator.Persistent);
        private NativeHashMap<int3, int> _layoutMap = new(kMaxLayouts, Allocator.Persistent);
        private NativeList<int> _usedIndices = new(kMaxLayouts, Allocator.Persistent);

        private int _maxLayoutCells = 0;
        
        public CellLayoutManager()
        {
            for (int i = 0; i < _cellLayouts.Length; i++)
            {
                _freeIndices.Add(i);
            }
        }

        public ref NativeArray<CellLayout> CellLayouts => ref _cellLayouts;


        public NativeList<int> UsedIndices => _usedIndices;

        public int TotalCount => _cellLayouts.Length;

        public int MaxLayoutCells => _maxLayoutCells;

        public void Dispose()
        {
            foreach (var index in _usedIndices)
            {
                _cellLayouts[index].Dispose();
            }

            _cellLayouts.Dispose();
            _freeIndices.Dispose();
            _usedIndices.Dispose();
            _layoutMap.Dispose();
        }

        public int RegisterLayout(float3 origin, float3 size, float cellSize, float maxDistance)
        {
            CellLayout cellLayout = new CellLayout(origin, size, cellSize, maxDistance);
            return RegisterLayout(cellLayout);
        }

        public int GetOrCreateLayout(float2 position, int layoutSize, int patchCount)
        {
            int3 key = default;
            key.x = (int) math.floor(position.x / layoutSize) * layoutSize;
            key.y = (int) math.floor(position.y / layoutSize) * layoutSize;
            key.z = patchCount;
            if (_layoutMap.TryGetValue(key, out var layoutId))
            {
                return layoutId;
            }

            float3 layoutPos = new float3(key.x, 0, key.y);
            float3 layoutSizeF = new float3(layoutSize, 0, layoutSize);

            // ReSharper disable once PossibleLossOfFraction
            CellLayout cellLayout = new CellLayout(layoutPos, layoutSizeF, layoutSize / patchCount, 
                BRGConstants.DefaultMaxDistance);
            return RegisterLayout(cellLayout);
        }

        private int RegisterLayout(in CellLayout layout)
        {
            if (_freeIndices.Length == 0)
                return -1;

            int last = _freeIndices.Length - 1;
            var index = _freeIndices[last];
            _freeIndices.RemoveAt(last);
            _cellLayouts[index] = layout;
            _usedIndices.Add(index);
            int3 key = GetKey(layout);
            _layoutMap.TryAdd(key, index);
            _maxLayoutCells = math.max(_maxLayoutCells, layout.Count * layout.Count);
            return index;
        }

        private int3 GetKey(in CellLayout layout)
        {
            int3 key = default;
            key.xy = (int2) layout.Origin;
            key.z = layout.Count;
            return key;
        }

        public void UnregisterLayout(int id)
        {
            if (id >= kMaxLayouts)
                return;
            CellLayout cellLayout = CellLayouts[id];

            int3 key = GetKey(cellLayout);
            _layoutMap.Remove(key);

            int index = _usedIndices.IndexOf(id);
            if (index == -1)
                return;

            _usedIndices.RemoveAtSwapBack(index);
            _freeIndices.Add(id);

            cellLayout.Dispose();
        }

        public void AddBounds(int layoutId, int cellIndex, in PackedMatrix packedMatrix, in PrototypePrefab prototypePrefab)
        {
            if (layoutId < 0 || layoutId >= _cellLayouts.Length)
                return;

            CellLayout cellLayout = _cellLayouts[layoutId];

            cellLayout.AddBounds(cellIndex, packedMatrix, prototypePrefab);
            _cellLayouts[layoutId] = cellLayout;
        }

        public void AddBounds(int layoutId, int index, in AABB bounds)
        {
            if (layoutId < 0 || layoutId >= _cellLayouts.Length)
                return;

            CellLayouts[layoutId].AddBounds(index, bounds);
        }
    }
}