using System;
using Unity.Mathematics;
using UnityEngine;

namespace FoliageBRG
{
    public class CellLayoutsGizmos : MonoBehaviour
    {
#if UNITY_EDITOR

        private static Color[] Colors =
        {
            new Color(0.85f, 0.12f, 0.54f), new Color(0.23f, 0.76f, 0.11f),
            new Color(0.98f, 0.65f, 0.04f), new Color(0.15f, 0.32f, 0.89f),
            new Color(0.44f, 0.91f, 0.72f), new Color(0.67f, 0.21f, 0.18f),
            new Color(0.09f, 0.55f, 0.33f), new Color(0.81f, 0.47f, 0.92f),
            new Color(0.36f, 0.14f, 0.66f), new Color(0.52f, 0.83f, 0.08f),
            new Color(0.94f, 0.28f, 0.41f), new Color(0.12f, 0.69f, 0.77f),
            new Color(0.73f, 0.39f, 0.05f), new Color(0.29f, 0.07f, 0.48f),
            new Color(0.58f, 0.96f, 0.22f), new Color(0.04f, 0.42f, 0.19f),
            new Color(0.88f, 0.74f, 0.51f), new Color(0.31f, 0.19f, 0.84f),
            new Color(0.62f, 0.03f, 0.37f), new Color(0.49f, 0.58f, 0.95f),
            new Color(0.21f, 0.87f, 0.63f), new Color(0.77f, 0.15f, 0.02f),
            new Color(0.18f, 0.34f, 0.56f), new Color(0.92f, 0.51f, 0.79f),
            new Color(0.06f, 0.71f, 0.13f), new Color(0.45f, 0.25f, 0.09f),
            new Color(0.69f, 0.81f, 0.44f), new Color(0.33f, 0.48f, 0.27f),
            new Color(0.11f, 0.09f, 0.72f), new Color(0.84f, 0.33f, 0.61f),
            new Color(0.55f, 0.67f, 0.08f), new Color(0.27f, 0.93f, 0.39f)
        };

        private void OnDrawGizmos()
        {
            FoliageBRGSystem brgSystem = FoliageBRGSystem.Instance;
            if (brgSystem.Disposed)
                return;

            CellLayoutManager cellLayoutManager = brgSystem.CellLayoutManager;

            if (cellLayoutManager == null || !cellLayoutManager.CellLayouts.IsCreated)
            {
                return;
            }


            for (var i1 = 0; i1 < cellLayoutManager.UsedIndices.Length; i1++)
            {
                var layoutIndex = cellLayoutManager.UsedIndices[i1];
                CellLayout cellLayout = cellLayoutManager.CellLayouts[layoutIndex];
                Gizmos.color = Colors[layoutIndex % Colors.Length];
                var style = new GUIStyle();
                style.normal.textColor = Gizmos.color;
                for (var i = 0; i < cellLayout.AABBs.Length; i++)
                {
                    var aabb = cellLayout.AABBs[i];
                    if (aabb.Extents.y <= float.Epsilon)
                        continue;
                    float3 size = aabb.Extents * 2;
                    Gizmos.DrawWireCube(aabb.Center, size);
                    UnityEditor.Handles.Label(aabb.Center, i.ToString(), style);
                }
            }
        }
#endif
    }
}