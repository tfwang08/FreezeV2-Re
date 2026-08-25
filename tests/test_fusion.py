import numpy as np

from freezev2.fusion import (
    apply_visual_pca,
    fit_visual_pca,
    fuse_visual_geometric,
)


def test_visual_pca_is_fit_on_query_and_reused_for_target():
    query = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    target = query + np.array([10.0, 0.0, 0.0])

    state = fit_visual_pca(query, output_dim=2)
    query_proj = apply_visual_pca(query, state)
    target_proj = apply_visual_pca(target, state)

    assert state.mean.shape == (3,)
    assert state.components.shape == (2, 3)
    np.testing.assert_allclose(query_proj.mean(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(target_proj.mean(axis=0), 0.0)


def test_fusion_normalizes_branches_before_concatenation():
    visual = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
        ]
    )
    geometric = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
            [8.0, 15.0],
            [7.0, 24.0],
        ]
    )

    state = fit_visual_pca(visual, output_dim=2)
    fused = fuse_visual_geometric(visual, geometric, state)

    assert fused.shape == (4, 4)
    np.testing.assert_allclose(np.linalg.norm(fused[:, :2], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(fused[:, 2:], axis=1), 1.0, atol=1e-6)


def test_fusion_requires_equal_visual_pca_and_geometric_dimensions():
    visual = np.eye(4)
    geometric = np.ones((4, 3))
    state = fit_visual_pca(visual, output_dim=2)

    try:
        fuse_visual_geometric(visual, geometric, state)
    except ValueError as exc:
        assert "same dimension" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch to raise")
