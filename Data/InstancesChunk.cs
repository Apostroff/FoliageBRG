namespace FoliageBRG
{
    public struct InstancesChunk
    {
        public readonly int IndicesStart;
        public readonly uint IndicesCount;
        public readonly int CellIndex;

        public InstancesChunk(int indicesStart, uint indicesCount, int cellIndex)
        {
            IndicesStart = indicesStart;
            IndicesCount = indicesCount;
            CellIndex = cellIndex;
        }
    }
}
