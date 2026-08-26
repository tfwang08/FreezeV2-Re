import numpy as np
import pytest


torch = pytest.importorskip("torch")

from freezev2.gedi_radius import radius_search


def test_radius_search_returns_flat_indices_and_row_splits():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    queries = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = torch.tensor([1.01, 1.01])

    indices, row_splits = radius_search(
        points,
        queries,
        radii,
        points_row_splits=torch.tensor([0, 3]),
        queries_row_splits=torch.tensor([0, 2]),
    )

    np.testing.assert_array_equal(indices.numpy(), [0, 1, 1, 2])
    np.testing.assert_array_equal(row_splits.numpy(), [0, 2, 4])
