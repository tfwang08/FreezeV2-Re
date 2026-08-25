import numpy as np
import pytest


torch = pytest.importorskip("torch")

from freezev2.gedi_compat import radius_search


def test_radius_search_matches_open3d_compact_layout():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    queries = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    radii = torch.tensor([1.1, 1.1], dtype=torch.float32)

    indices, row_splits, distances = radius_search(
        points,
        queries,
        radii,
        points_row_splits=torch.tensor([0, 3]),
        queries_row_splits=torch.tensor([0, 2]),
    )

    assert indices.tolist() == [0, 1, 1, 2]
    assert row_splits.tolist() == [0, 2, 4]
    assert distances.numel() == 0


def test_radius_search_can_return_squared_l2_distances():
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    queries = torch.tensor([[1.0, 0.0, 0.0]])
    radii = torch.tensor([2.0])

    indices, row_splits, distances = radius_search(
        points,
        queries,
        radii,
        return_distances=True,
    )

    assert indices.tolist() == [0, 1]
    assert row_splits.tolist() == [0, 2]
    np.testing.assert_allclose(distances.numpy(), [1.0, 1.0])
