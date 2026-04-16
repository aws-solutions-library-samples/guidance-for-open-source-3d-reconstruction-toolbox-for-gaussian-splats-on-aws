#!/usr/bin/env python3
"""
Generate a convex hull collision mesh from a Gaussian splat PLY file.
Reads XYZ positions from the splat, computes a convex hull, and writes
a triangle mesh PLY suitable for use as a USDZ collision geometry.
"""
import argparse
import struct
import numpy as np


def read_ply_positions(ply_path: str) -> np.ndarray:
    """Read only x,y,z vertex positions from a binary or ASCII PLY."""
    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        num_vertices = 0
        is_binary_little = False
        is_binary_big = False
        props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith('element') and not line.startswith('element vertex'):
                in_vertex = False
            elif line.startswith('property') and in_vertex:
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)
            elif line == 'format binary_little_endian 1.0':
                is_binary_little = True
            elif line == 'format binary_big_endian 1.0':
                is_binary_big = True

        type_map = {
            'float': ('f', 4), 'float32': ('f', 4),
            'double': ('d', 8), 'float64': ('d', 8),
            'int': ('i', 4), 'int32': ('i', 4),
            'uint': ('I', 4), 'uint32': ('I', 4),
            'short': ('h', 2), 'int16': ('h', 2),
            'ushort': ('H', 2), 'uint16': ('H', 2),
            'uchar': ('B', 1), 'uint8': ('B', 1),
            'char': ('b', 1), 'int8': ('b', 1),
        }

        if is_binary_little or is_binary_big:
            endian = '<' if is_binary_little else '>'
            fmt = endian + ''.join(type_map[t][0] for t, _ in props)
            row_size = struct.calcsize(fmt)
            x_idx = next(i for i, (_, n) in enumerate(props) if n == 'x')
            y_idx = next(i for i, (_, n) in enumerate(props) if n == 'y')
            z_idx = next(i for i, (_, n) in enumerate(props) if n == 'z')
            data = f.read(num_vertices * row_size)
            rows = struct.iter_unpack(fmt, data)
            positions = np.array([[r[x_idx], r[y_idx], r[z_idx]] for r in rows], dtype=np.float32)
        else:
            # ASCII
            positions = []
            name_to_idx = {n: i for i, (_, n) in enumerate(props)}
            xi, yi, zi = name_to_idx['x'], name_to_idx['y'], name_to_idx['z']
            for _ in range(num_vertices):
                vals = f.readline().decode('ascii').split()
                positions.append([float(vals[xi]), float(vals[yi]), float(vals[zi])])
            positions = np.array(positions, dtype=np.float32)

    return positions


def write_mesh_ply(path: str, vertices: np.ndarray, faces: np.ndarray):
    """Write a triangle mesh as binary little-endian PLY."""
    with open(path, 'wb') as f:
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"element face {len(faces)}\n"
            f"property list uchar int vertex_indices\n"
            f"end_header\n"
        )
        f.write(header.encode('ascii'))
        f.write(vertices.astype(np.float32).tobytes())
        for face in faces:
            f.write(struct.pack('B', 3))
            f.write(struct.pack('<iii', *face))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input splat PLY')
    parser.add_argument('-o', '--output', required=True, help='Output mesh PLY')
    parser.add_argument('--max-points', type=int, default=5000,
                        help='Subsample to this many points before hull (default: 5000)')
    args = parser.parse_args()

    import trimesh

    print(f"Reading positions from {args.input}")
    positions = read_ply_positions(args.input)
    print(f"  {len(positions)} splat points read")

    # Subsample to keep convex hull fast and small
    if len(positions) > args.max_points:
        idx = np.random.choice(len(positions), args.max_points, replace=False)
        positions = positions[idx]

    print(f"Computing convex hull from {len(positions)} points...")
    cloud = trimesh.PointCloud(positions)
    hull = cloud.convex_hull
    print(f"  Hull: {len(hull.vertices)} vertices, {len(hull.faces)} faces")

    write_mesh_ply(args.output, hull.vertices, hull.faces)
    print(f"Collision mesh written to {args.output}")


if __name__ == '__main__':
    main()
