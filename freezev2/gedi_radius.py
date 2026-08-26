from __future__ import annotations

from types import SimpleNamespace


def radius_search(
    points,
    queries,
    radii,
    *,
    points_row_splits,
    queries_row_splits,
    chunk_size: int = 256,
):
    """Single-batch fixed-radius search matching the GeDi Open3D call shape.

    GeDi passes CPU torch tensors here. The implementation deliberately stays
    on CPU so it is independent of the Open3D/PyTorch binary ABI and therefore
    works with both the Ampere reproduction stack and Blackwell PyTorch builds.
    """
    import torch

    points = torch.as_tensor(points, dtype=torch.float32, device="cpu")
    queries = torch.as_tensor(queries, dtype=torch.float32, device="cpu")
    radii = torch.as_tensor(radii, dtype=torch.float32, device="cpu").reshape(-1)
    points_row_splits = torch.as_tensor(
        points_row_splits, dtype=torch.long, device="cpu"
    )
    queries_row_splits = torch.as_tensor(
        queries_row_splits, dtype=torch.long, device="cpu"
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape Nx3")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must have shape Mx3")
    if radii.shape != (len(queries),):
        raise ValueError("radii must contain one radius per query")
    if torch.any(radii < 0):
        raise ValueError("radii must be non-negative")
    if points_row_splits.tolist() != [0, len(points)]:
        raise ValueError("GeDi radius search currently supports one point-cloud batch")
    if queries_row_splits.tolist() != [0, len(queries)]:
        raise ValueError("GeDi radius search currently supports one query batch")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat_indices = []
    row_splits = [0]
    for start in range(0, len(queries), int(chunk_size)):
        stop = min(start + int(chunk_size), len(queries))
        distances = torch.cdist(queries[start:stop], points)
        within = distances <= radii[start:stop, None]
        pairs = torch.nonzero(within, as_tuple=False)
        counts = (
            torch.bincount(pairs[:, 0], minlength=stop - start)
            if len(pairs)
            else torch.zeros(stop - start, dtype=torch.long)
        )
        if len(pairs):
            flat_indices.append(pairs[:, 1].to(torch.long))
        for count in counts.tolist():
            row_splits.append(row_splits[-1] + int(count))

    indices = (
        torch.cat(flat_indices)
        if flat_indices
        else torch.empty((0,), dtype=torch.long)
    )
    return indices, torch.tensor(row_splits, dtype=torch.long)


ml3d = SimpleNamespace(ops=SimpleNamespace(radius_search=radius_search))
