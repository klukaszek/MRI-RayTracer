from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_ply_ascii(path: Path, max_faces: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal ASCII PLY loader that reads vertices (x,y,z) and triangular faces.
    Returns:
      vertices: float32 array [N,3]
      triangles: uint32 array [M,3]
    If max_faces is provided, caps the number of faces loaded (after filtering for triangles).
    """
    with open(path, "r", encoding="utf-8") as f:
        # Parse header
        line = f.readline()
        if not line.startswith("ply"):
            raise ValueError("Not a PLY file")
        vertex_count = 0
        face_count = 0
        header_ended = False
        properties = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading header")
            line = line.strip()
            if line == "end_header":
                header_ended = True
                break
            if line.startswith("element vertex"):
                parts = line.split()
                vertex_count = int(parts[-1])
            elif line.startswith("element face"):
                parts = line.split()
                face_count = int(parts[-1])
            elif line.startswith("property"):
                properties.append(line)

        if not header_ended:
            raise ValueError("Invalid PLY header")

        # Read vertices
        verts = np.zeros((vertex_count, 3), dtype=np.float32)
        for i in range(vertex_count):
            parts = f.readline().strip().split()
            if len(parts) < 3:
                raise ValueError("Malformed vertex line")
            verts[i, 0] = float(parts[0])
            verts[i, 1] = float(parts[1])
            verts[i, 2] = float(parts[2])

        # Read faces
        tris = []
        loaded = 0
        limit = max_faces if max_faces is not None else face_count
        for _ in range(face_count):
            if loaded >= limit:
                # Skip the rest of lines to keep file pointer clean (optional)
                f.readline()
                continue
            parts = f.readline().strip().split()
            if not parts:
                continue
            n = int(parts[0])
            if n != 3:
                # Skip non-triangle faces
                continue
            if len(parts) < 4:
                continue
            i0 = int(parts[1]); i1 = int(parts[2]); i2 = int(parts[3])
            tris.append((i0, i1, i2))
            loaded += 1

        if not tris:
            raise ValueError("No triangular faces found in PLY")

        tri_arr = np.array(tris, dtype=np.uint32)
        return verts, tri_arr

