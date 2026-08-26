import csv
import numpy as np
from freezev2.geometry import backproject_depth, kabsch
from freezev2.matching import topk_cosine_matches
from freezev2.pose import _score_hypothesis, feature_aware_ransac, sparse_grid_pixels
from freezev2.pipeline import estimate_pose_from_features
from freezev2.bop import write_bop_csv


def test_kabsch_recovers_known_rigid_transform():
    src = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    a = np.deg2rad(30.0)
    R = np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    t = np.array([0.2,-0.3,1.4])
    dst = src @ R.T + t
    Rh, th = kabsch(src, dst)
    assert np.allclose(Rh, R, atol=1e-7)
    assert np.allclose(th, t, atol=1e-7)


def test_backproject_depth():
    depth = np.array([[0.,2.],[3.,4.]], dtype=np.float32)
    mask = np.array([[False,True],[False,False]])
    K = np.array([[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]])
    pts, uv = backproject_depth(depth, K, mask)
    assert np.allclose(pts[0], [1.,0.,2.])
    assert np.array_equal(uv[0], [1,0])


def test_topk_cosine_matching():
    target = np.array([[1.,0.],[0.,1.]])
    query = np.array([[1.,0.],[0.8,0.2],[0.,1.]])
    idx, sim = topk_cosine_matches(target, query, k=2)
    assert idx.tolist() == [[0,1],[2,1]]
    assert np.all(sim[:,0] >= sim[:,1])


def test_sparse_grid_stays_inside_mask():
    mask = np.zeros((100,200), dtype=bool)
    mask[20:80,40:160] = True
    uv = sparse_grid_pixels(mask, 16)
    assert 100 < len(uv) <= 256
    assert np.all(mask[uv[:,1], uv[:,0]])


def test_feature_aware_score_sums_all_inlier_correspondences():
    pose = np.eye(4, dtype=np.float64)
    query_points = np.zeros((2, 3), dtype=np.float64)
    target_points = np.zeros((1, 3), dtype=np.float64)
    query_features = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(0.75)],
    ])
    target_features = np.array([[1.0, 0.0]])
    candidate_query_indices = np.array([[0, 1]], dtype=np.int64)

    score = _score_hypothesis(
        pose,
        query_points,
        query_features,
        target_points,
        target_features,
        candidate_query_indices,
        inlier_threshold=1.0,
    )

    # FreeZeV2 Eq. (5): sum the cosine of every inlier correspondence,
    # then divide by the number of sparse target points. Both correspondences
    # are exact geometric inliers here, with cosine 1.0 and 0.5.
    np.testing.assert_allclose(score, 1.5)


def test_feature_aware_pipeline_recovers_pose():
    rng = np.random.default_rng(1)
    qpts = rng.normal(size=(20,3))
    qfeat = rng.normal(size=(20,8))
    qfeat /= np.linalg.norm(qfeat, axis=1, keepdims=True)
    R = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    t = np.array([0.1,0.2,0.7])
    ids = np.arange(10)
    tpts = qpts[ids] @ R.T + t
    pose, score = estimate_pose_from_features(qpts, qfeat, tpts, qfeat[ids], 2.0, k=3, iterations=1000, seed=2)
    assert score > 0.9
    assert np.allclose(pose[:3,:3], R, atol=1e-5)
    assert np.allclose(pose[:3,3], t, atol=1e-5)


def test_bop_writer(tmp_path):
    pose = np.eye(4); pose[:3,3] = [10.,20.,30.]
    out = tmp_path / "pred.csv"
    write_bop_csv(out, [{"scene_id":1,"im_id":2,"obj_id":3,"score":0.75,"pose":pose,"time":1.25}])
    row = list(csv.DictReader(out.open()))[0]
    assert row["R"] == "1 0 0 0 1 0 0 0 1"
    assert row["t"] == "10 20 30"
