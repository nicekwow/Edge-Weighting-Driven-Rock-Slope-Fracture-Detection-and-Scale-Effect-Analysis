"""Measure axial fracture-trace directions on the image plane.

Directions are image-plane orientations, not geologic strikes. Each trace is
skeletonized, separated at junctions, and sampled by arc length. The rose bins
describe the normalized length distribution, not recovered total trace length.
"""

import math

import numpy as np
from skimage.morphology import skeletonize


def branches(mask):
    """Return paths between skeleton endpoints/junctions, retaining loops.

    Diagonal adjacency is suppressed when an orthogonal bridge already joins
    the same pixels. This avoids counting a corner connection twice.
    """
    skeleton = skeletonize(np.ascontiguousarray(mask))
    coords = list(zip(*np.where(skeleton)))  # (row, column) per skeleton pixel
    index = {point: i for i, point in enumerate(coords)}
    adjacency = [[] for _ in coords]
    edges = {}

    for i, (y, x) in enumerate(coords):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            j = index.get((y + dy, x + dx))
            if j is None:
                continue
            if dx and dy and ((y, x + dx) in index or (y + dy, x) in index):
                continue
            adjacency[i].append(j)
            adjacency[j].append(i)
            edges[(min(i, j), max(i, j))] = math.hypot(dx, dy)

    visited = set()
    paths = []

    def walk(start, neighbor):
        path = [start]
        previous, current = start, neighbor
        while True:
            edge = tuple(sorted((previous, current)))
            if edge in visited:
                break
            visited.add(edge)
            path.append(current)
            if len(adjacency[current]) != 2:
                break
            following = (adjacency[current][0] if adjacency[current][1] == previous
                         else adjacency[current][1])
            previous, current = current, following
        return np.asarray([coords[i] for i in path], dtype=float)

    # Open paths start at vertices whose degree is not two.
    for start, neighbors in enumerate(adjacency):
        if len(neighbors) != 2:
            for neighbor in neighbors:
                if tuple(sorted((start, neighbor))) not in visited:
                    paths.append(walk(start, neighbor))

    # Edges left over belong to closed loops with no endpoints or junctions.
    for start, neighbor in edges:
        if (start, neighbor) not in visited:
            paths.append(walk(start, neighbor))
    assert len(visited) == len(edges)
    return paths, sum(edges.values())


def orientations(paths, span=20):
    """Return segment directions, length weights, and 18 axial rose bins.

    Paths shorter than 10 pixels are excluded. Remaining paths are divided
    into approximately ``span``-pixel arcs. The chord between each arc's ends
    sets its direction; the arc length, rather than chord length, is its weight.
    Zero degrees denotes image vertical, with directions wrapping at 180.
    """
    angles, weights = [], []
    short_length = degenerate_length = 0.0
    for path in paths:
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        length = arc[-1]
        if length < 10:
            short_length += length
            continue
        count = max(1, round(length / span))
        if np.array_equal(path[0], path[-1]):
            count = max(count, 3)
        cuts = np.linspace(0, length, count + 1)
        ys = np.interp(cuts, arc, path[:, 0])
        xs = np.interp(cuts, arc, path[:, 1])
        for dy, dx, weight in zip(np.diff(ys), np.diff(xs), np.diff(cuts)):
            if math.hypot(dx, dy) < 1e-6:
                degenerate_length += weight
                continue
            angles.append(math.degrees(math.atan2(dx, -dy)) % 180)
            weights.append(weight)

    angles = np.asarray(angles)
    weights = np.asarray(weights)
    # Bins are centered at 0, 10, ..., 170 degrees; +5 handles wraparound.
    indices = np.floor(((angles + 5) % 180) / 10).astype(int)
    histogram = np.bincount(indices, weights=weights, minlength=18)
    if histogram.sum():
        histogram /= histogram.sum()
    return angles, weights, histogram, short_length, degenerate_length


def validate():
    """Check basic directions and exact coverage of skeleton graph edges."""
    for kind, expected in (("vertical", 0), ("horizontal", 90), ("diagonal", 135)):
        mask = np.zeros((80, 80), bool)
        if kind == "vertical":
            mask[10:70, 40] = True
        elif kind == "horizontal":
            mask[40, 10:70] = True
        else:
            values = np.arange(10, 70)
            mask[values, values] = True
        paths, length = branches(mask)
        angles, weights, _, _, _ = orientations(paths)
        assert np.allclose(angles, expected), (kind, angles)
        assert abs(weights.sum() - length) < 1e-7

    mask = np.zeros((40, 40), bool)
    mask[5:35, 20] = True
    mask[20, 5:35] = True
    paths, length = branches(mask)
    path_length = sum(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                      for path in paths)
    assert abs(path_length - length) < 1e-8


validate()
