#!/usr/bin/env python3
"""Generate binary vertex, index, and transform payloads + draw.json for the
HV-01 graphics exemplar (triga-hello-voxel-shaders.fab).

The locked contract defines a unit cube with 8 corners and 12 indexed
triangles (36 indices). Vertex data is structure-of-arrays: one buffer
for positions, one for colors (12 bytes per vertex each).
"""

import json
import struct
import sys
from pathlib import Path

# Cube corners (8 vertices)
CORNERS = [
    (0.0, 0.0, 0.0),  # 0
    (1.0, 0.0, 0.0),  # 1
    (1.0, 1.0, 0.0),  # 2
    (0.0, 1.0, 0.0),  # 3
    (0.0, 0.0, 1.0),  # 4
    (1.0, 0.0, 1.0),  # 5
    (1.0, 1.0, 1.0),  # 6
    (0.0, 1.0, 1.0),  # 7
]

CORNER_COLORS = [
    (1.0, 0.0, 0.0),  # 0
    (0.0, 1.0, 0.0),  # 1
    (0.0, 0.0, 1.0),  # 2
    (1.0, 1.0, 0.0),  # 3
    (1.0, 0.0, 1.0),  # 4
    (0.0, 1.0, 1.0),  # 5
    (1.0, 1.0, 1.0),  # 6
    (0.2, 0.2, 0.2),  # 7
]

# 12 triangles = 36 indices
INDICES = [
    0, 1, 2, 0, 2, 3,    # front
    4, 6, 5, 4, 7, 6,    # back
    0, 4, 5, 0, 5, 1,    # bottom
    3, 2, 6, 3, 6, 7,    # top
    1, 5, 6, 1, 6, 2,    # right
    0, 3, 7, 0, 7, 4,    # left
]


def expand_vertices():
    """Expand indexed geometry to 36 structure-of-arrays vertices."""
    positions = []
    colors = []
    for idx in INDICES:
        px, py, pz = CORNERS[idx]
        cx, cy, cz = CORNER_COLORS[idx]
        positions.extend([px, py, pz])
        colors.extend([cx, cy, cz])
    return positions, colors


def write_f32_array(path, values):
    """Write a list of f32 values as binary."""
    data = struct.pack(f"<{len(values)}f", *values)
    Path(path).write_bytes(data)


def write_u32_array(path, values):
    """Write a list of u32 values as binary."""
    data = struct.pack(f"<{len(values)}I", *values)
    Path(path).write_bytes(data)


def identity_matrix4():
    """Return 16 f32 values for a column-major identity 4x4 matrix."""
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def generate_transform(output_dir):
    """Generate 256-byte transform payload (64 f32):
    model matrix (16 f32) + view-projection matrix (16 f32) + padding (32 f32).
    """
    model = identity_matrix4()
    view_proj = identity_matrix4()
    padding = [0.0] * 32
    values = model + view_proj + padding
    write_f32_array(output_dir / "graphics-transform.bin", values)


def generate_draw_manifest(output_dir):
    manifest = {
        "index_format": "uint32",
        "instance_count": 1,
        "base_vertex": 0,
        "first_index": 0,
        "index_count": 36,
    }
    (output_dir / "draw.json").write_text(json.dumps(manifest) + "\n")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    positions, colors = expand_vertices()
    write_f32_array(output_dir / "graphics-vertex-positions.bin", positions)
    write_f32_array(output_dir / "graphics-vertex-colors.bin", colors)
    write_u32_array(output_dir / "graphics-indices.bin", INDICES)
    generate_transform(output_dir)
    generate_draw_manifest(output_dir)

    print(f"generated {output_dir / 'graphics-vertex-positions.bin'} ({len(positions) * 4} bytes)")
    print(f"generated {output_dir / 'graphics-vertex-colors.bin'} ({len(colors) * 4} bytes)")
    print(f"generated {output_dir / 'graphics-indices.bin'} ({len(INDICES) * 4} bytes)")
    print(f"generated {output_dir / 'graphics-transform.bin'} (256 bytes)")
    print(f"generated {output_dir / 'draw.json'}")


if __name__ == "__main__":
    main()
