from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BVH:
    nodes: np.ndarray  # float32 [N, 8] -> min(3), max(3), leftFirst(1), triCount(1)
    tris: np.ndarray   # uint32 [M, 3]
    vert_pos: np.ndarray  # float32 [V, 3]


def build_bvh(vert_pos: np.ndarray, tris: np.ndarray, max_leaf_tris: int = 4) -> BVH:
    """
    Very small SAH-less BVH: recursive median split along largest axis of centroid bounds.
    Packs nodes as:
      [minx, miny, minz, maxx, maxy, maxz, leftFirst_or_start, triCount_or_count]
    - For inner: leftFirst = index of left child, triCount = 0
    - For leaf: leftFirst = start index into a compacted index buffer; triCount = count
    We compact triangles for leaves in build order into a new tri list.
    """
    V = vert_pos.astype(np.float32, copy=False)
    T = tris.astype(np.uint32, copy=False)

    tri_centroids = V[T].mean(axis=1)
    tri_bounds_min = V[T].min(axis=1)
    tri_bounds_max = V[T].max(axis=1)

    nodes_min = []
    nodes_max = []
    left_first = []
    tri_count = []
    leaf_tri_indices = []  # flattened tri indices in leaf order

    def emit_node(bmin, bmax, lf, cnt):
        nodes_min.append(bmin)
        nodes_max.append(bmax)
        left_first.append(lf)
        tri_count.append(cnt)
        return len(left_first) - 1

    def rec(tri_ids: np.ndarray) -> int:
        bmin = tri_bounds_min[tri_ids].min(axis=0)
        bmax = tri_bounds_max[tri_ids].max(axis=0)
        if len(tri_ids) <= max_leaf_tris:
            start = len(leaf_tri_indices)
            for t in tri_ids:
                leaf_tri_indices.append(int(t))
            node_idx = emit_node(bmin, bmax, start, len(tri_ids))
            return node_idx
        # split by largest axis of centroid bounds
        cmin = tri_centroids[tri_ids].min(axis=0)
        cmax = tri_centroids[tri_ids].max(axis=0)
        axis = int(np.argmax(cmax - cmin))
        order = np.argsort(tri_centroids[tri_ids, axis])
        mid = len(tri_ids) // 2
        left_ids = tri_ids[order[:mid]]
        right_ids = tri_ids[order[mid:]]
        # placeholder inner
        node_idx = emit_node(bmin, bmax, 0, 0)
        l = rec(left_ids)
        r = rec(right_ids)
        # set left child index, and encode right child index as negative value to mark inner node
        left_first[node_idx] = l
        tri_count[node_idx] = -(r + 1)  # negative => inner; decode as r = -val - 1
        return node_idx

    root = rec(np.arange(len(T), dtype=np.int32))

    # Pack nodes as float layout that matches BVHNode in shader
    nodes = np.zeros((len(left_first), 8), dtype=np.float32)
    nodes[:, 0:3] = np.array(nodes_min, dtype=np.float32)     # bmin.xyz
    nodes[:, 3] = np.array([m[0] for m in nodes_max], dtype=np.float32)  # bmax.x
    nodes[:, 4:6] = np.array([[m[1], m[2]] for m in nodes_max], dtype=np.float32)  # bmax.yz
    # Store as numeric floats (avoid bitcast semantics differences across backends)
    nodes[:, 6] = np.array(left_first, dtype=np.float32)
    nodes[:, 7] = np.array(tri_count, dtype=np.float32)

    # Compact tris
    compact_tris = tris[np.array(leaf_tri_indices, dtype=np.int32)]
    return BVH(nodes=nodes, tris=compact_tris, vert_pos=V)
