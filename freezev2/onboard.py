from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import tempfile

import numpy as np

from .geometry import project_points


@dataclass(frozen=True)
class CameraPose:
    """OpenCV-style model-to-camera pose (+x right, +y down, +z forward)."""

    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    size: int
    direction: np.ndarray


@dataclass(frozen=True)
class Template:
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    camera: CameraPose


def _mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices must be Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh faces must be Mx3 triangles")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("mesh must contain vertices and triangle faces")
    return vertices, faces


def load_mesh(path: str | Path):
    """Load a BOP mesh lazily through trimesh."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Install onboarding dependencies with: pip install -e '.[onboard]'") from exc

    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.vertices) and len(g.faces)]
        if not geometries:
            raise ValueError(f"No triangle mesh found in {path}")
        mesh = trimesh.util.concatenate(geometries)
    _mesh_arrays(mesh)
    return mesh


def _uniform_surface_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("mesh has zero surface area")

    face_ids = rng.choice(len(faces), size=int(count), p=areas / total)
    selected = tri[face_ids]
    r1 = np.sqrt(rng.random(int(count)))
    r2 = rng.random(int(count))
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    points = (
        a[:, None] * selected[:, 0]
        + b[:, None] * selected[:, 1]
        + c[:, None] * selected[:, 2]
    )
    return points, total


def _poisson_reject(points: np.ndarray, radius: float, count: int) -> np.ndarray:
    """Greedily enforce a Euclidean Poisson-disk radius with a 3D hash grid."""
    if radius <= 0:
        return np.asarray(points[:count], dtype=np.float64)
    inv = 1.0 / radius
    radius_sq = radius * radius
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )
    grid: dict[tuple[int, int, int], list[np.ndarray]] = {}
    accepted: list[np.ndarray] = []
    for point in points:
        key_arr = np.floor(point * inv).astype(np.int64)
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        keep = True
        for dx, dy, dz in offsets:
            neighbour = (key[0] + dx, key[1] + dy, key[2] + dz)
            for other in grid.get(neighbour, ()):
                delta = point - other
                if float(delta @ delta) < radius_sq:
                    keep = False
                    break
            if not keep:
                break
        if keep:
            accepted.append(point)
            grid.setdefault(key, []).append(point)
            if len(accepted) == count:
                break
    return np.asarray(accepted, dtype=np.float64)


def sample_query_points(mesh, n: int = 5000, seed: int = 0) -> np.ndarray:
    """Sample exactly ``n`` surface points with Poisson-disk spacing.

    The paper specifies Poisson disk sampling but does not publish its mesh
    implementation or ``N_Q^raw``. We therefore keep the required property
    explicit: area-uniform surface candidates followed by deterministic
    minimum-distance rejection.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    vertices, faces = _mesh_arrays(mesh)
    rng = np.random.default_rng(seed)
    candidates, area = _uniform_surface_candidates(vertices, faces, int(n) * 8, rng)

    radius = math.sqrt(area / (3.0 * int(n)))
    for _ in range(16):
        sampled = _poisson_reject(candidates, radius, int(n))
        if len(sampled) == int(n):
            return sampled
        radius *= 0.9

    sampled = _poisson_reject(candidates, 0.0, int(n))
    if len(sampled) != int(n):
        raise RuntimeError(f"Could only sample {len(sampled)}/{n} surface points")
    return sampled


def _base_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Regular icosahedron oriented like CNOS/Blender, with vertices at the poles."""
    z = 1.0 / math.sqrt(5.0)
    r = 2.0 / math.sqrt(5.0)
    vertices = [[0.0, 0.0, -1.0]]
    for k in range(5):
        angle = math.radians(36.0 + 72.0 * k)
        vertices.append([r * math.cos(angle), r * math.sin(angle), -z])
    for k in range(5):
        angle = math.radians(72.0 * k)
        vertices.append([r * math.cos(angle), r * math.sin(angle), z])
    vertices.append([0.0, 0.0, 1.0])

    faces: list[tuple[int, int, int]] = []
    for k in range(5):
        faces.append((0, 1 + k, 1 + ((k + 1) % 5)))
    for k in range(5):
        lower = 1 + k
        lower_next = 1 + ((k + 1) % 5)
        upper = 6 + k
        upper_next = 6 + ((k + 1) % 5)
        faces.append((lower, upper, upper_next))
        faces.append((lower, upper_next, lower_next))
    for k in range(5):
        faces.append((11, 6 + ((k + 1) % 5), 6 + k))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _subdivide_icosphere(vertices: np.ndarray, faces: np.ndarray):
    out_vertices = vertices.tolist()
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key not in midpoint_cache:
            point = vertices[key[0]] + vertices[key[1]]
            point = point / np.linalg.norm(point)
            midpoint_cache[key] = len(out_vertices)
            out_vertices.append(point.tolist())
        return midpoint_cache[key]

    out_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        out_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    return np.asarray(out_vertices, dtype=np.float64), np.asarray(out_faces, dtype=np.int64)


def _cnos_directions(n: int) -> np.ndarray:
    subdivision_by_count = {12: 0, 42: 1, 162: 2, 642: 3}
    if n not in subdivision_by_count:
        raise ValueError("n must be one of 12, 42, 162, 642 for the CNOS icosphere hierarchy")
    vertices, faces = _base_icosahedron()
    for _ in range(subdivision_by_count[n]):
        vertices, faces = _subdivide_icosphere(vertices, faces)

    elevation = np.arctan2(vertices[:, 2], np.linalg.norm(vertices[:, :2], axis=1))
    azimuth = np.arctan2(vertices[:, 0], vertices[:, 1])
    order = np.lexsort((azimuth, elevation))
    return vertices[order]


def _camera_rotation(direction: np.ndarray) -> np.ndarray:
    """CNOS look-at convention, returned as model/world-to-camera rotation."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    forward = -direction
    tmp = np.array([0.0, 0.0, -1.0])
    if min(np.linalg.norm(direction - tmp), np.linalg.norm(direction + tmp)) < 1e-3:
        tmp = np.array([0.0, -1.0, 0.0])
    right = np.cross(tmp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    camera_to_world = np.stack((right, up, forward), axis=1)
    return camera_to_world.T


def make_template_cameras(
    n: int = 162,
    size: int = 480,
    focal: float | None = None,
) -> list[CameraPose]:
    """Create the CNOS icosphere viewpoints used by FreeZeV2."""
    if size <= 0:
        raise ValueError("size must be positive")
    focal = float(size if focal is None else focal)
    c = 0.5 * (size - 1)
    K = np.array([[focal, 0.0, c], [0.0, focal, c], [0.0, 0.0, 1.0]])
    cameras = []
    for direction in _cnos_directions(int(n)):
        R = _camera_rotation(direction)
        t = -R @ direction
        cameras.append(CameraPose(R=R, t=t, K=K.copy(), size=int(size), direction=direction.copy()))
    return cameras


def fit_camera_to_mesh(mesh, camera: CameraPose, target_fill: float = 0.5) -> CameraPose:
    """Choose camera distance so projected mesh bbox spans target_fill of the frame."""
    if not (0.0 < target_fill < 1.0):
        raise ValueError("target_fill must be in (0, 1)")
    vertices, _ = _mesh_arrays(mesh)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    offsets = vertices - center
    radius = float(np.max(np.linalg.norm(offsets, axis=1)))
    if radius <= 0:
        raise ValueError("mesh has zero extent")

    R = camera.R
    rotated = offsets @ R.T
    target_span = target_fill * camera.size

    def span_at(distance: float) -> float:
        z = rotated[:, 2] + distance
        if np.any(z <= 0):
            return float("inf")
        u = camera.K[0, 0] * rotated[:, 0] / z + camera.K[0, 2]
        v = camera.K[1, 1] * rotated[:, 1] / z + camera.K[1, 2]
        return max(float(np.ptp(u)), float(np.ptp(v)))

    low = max(radius * 1.001, 1e-9)
    high = max(radius * 4.0, 1.0)
    while span_at(high) > target_span:
        high *= 2.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        if span_at(mid) > target_span:
            low = mid
        else:
            high = mid
    distance = high
    camera_position = center + camera.direction * distance
    t = -R @ camera_position
    return replace(camera, t=t)


def _prepare_renderer_mesh(mesh):
    """Return a trimesh, a PLY path for BOP Toolkit, and optional temp storage."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Install onboarding dependencies with: pip install -e '.[onboard]'") from exc

    if isinstance(mesh, (str, Path)):
        path = Path(mesh)
        return load_mesh(path), path, None

    vertices, faces = _mesh_arrays(mesh)
    if isinstance(mesh, trimesh.Trimesh):
        tri_mesh = mesh
    else:
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    temp_dir = tempfile.TemporaryDirectory(prefix="freezev2-render-")
    path = Path(temp_dir.name) / "mesh.ply"
    tri_mesh.export(path)
    return tri_mesh, path, temp_dir


def render_templates(
    mesh,
    cameras: list[CameraPose],
    size: int = 480,
    target_fill: float = 0.5,
) -> list[Template]:
    """Render RGB/depth templates with the official BOP VisPy renderer.

    BOP's VisPy backend uses the same headless EGL path as the Stage-1 evaluator
    and accepts OpenCV-style ``R/t/K`` directly. This avoids pyrender's EGL
    device enumeration, which is incompatible with software Mesa/llvmpipe on
    machines where ``eglQueryDevicesEXT`` reports no devices.
    """
    try:
        from bop_toolkit_lib.rendering import renderer as bop_renderer
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned BOP Toolkit before rendering CAD templates"
        ) from exc

    mesh_obj, mesh_path, temp_dir = _prepare_renderer_mesh(mesh)
    renderer = bop_renderer.create_renderer(
        int(size),
        int(size),
        renderer_type="vispy",
        mode="rgb+depth",
        shading="flat",
        bg_color=(0.0, 0.0, 0.0, 0.0),
    )
    obj_id = 1
    renderer.set_light_cam_pos((0.0, 0.0, 0.0))
    renderer.set_light_ambient_weight(0.02)
    renderer.add_object(obj_id, str(mesh_path))
    templates: list[Template] = []

    try:
        for base_camera in cameras:
            if base_camera.size != size:
                scale = size / float(base_camera.size)
                K = base_camera.K.copy()
                K[0, :] *= scale
                K[1, :] *= scale
                K[2, :] = [0.0, 0.0, 1.0]
                base_camera = replace(base_camera, K=K, size=int(size))

            camera = fit_camera_to_mesh(mesh_obj, base_camera, target_fill=target_fill)
            output = renderer.render_object(
                obj_id=obj_id,
                R=np.asarray(camera.R, dtype=np.float64),
                t=np.asarray(camera.t, dtype=np.float64).reshape(3, 1),
                fx=float(camera.K[0, 0]),
                fy=float(camera.K[1, 1]),
                cx=float(camera.K[0, 2]),
                cy=float(camera.K[1, 2]),
            )
            rgb = np.asarray(output["rgb"], dtype=np.uint8)
            depth = np.asarray(output["depth"], dtype=np.float32)
            templates.append(
                Template(rgb=rgb, depth=depth, mask=depth > 0, camera=camera)
            )
    finally:
        renderer.remove_object(obj_id)
        if temp_dir is not None:
            temp_dir.cleanup()

    return templates


def map_visible_pixels_to_query_points(
    query_points: np.ndarray,
    depth: np.ndarray,
    camera: CameraPose,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return query-point indices and pixels whose projected depth matches the render."""
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")
    query_points = np.asarray(query_points, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    uv, z = project_points(query_points, camera.K, camera.R, camera.t)
    finite = np.isfinite(uv).all(axis=1)
    pixels = np.zeros((len(uv), 2), dtype=np.int64)
    pixels[finite] = np.rint(uv[finite]).astype(np.int64)
    h, w = depth.shape[:2]
    valid = (
        finite
        & (z > 0)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return ids, np.empty((0, 2), dtype=np.int64)
    p = pixels[ids]
    rendered_z = depth[p[:, 1], p[:, 0]]
    visible = (rendered_z > 0) & (np.abs(rendered_z - z[ids]) <= depth_tolerance)
    return ids[visible], p[visible]


def save_onboarding_cache(
    path: str | Path,
    query_points: np.ndarray,
    cameras: list[CameraPose],
    **extra_arrays,
) -> Path:
    """Save one compact NPZ per object; later stages can append descriptors/PCA state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_points": np.asarray(query_points, dtype=np.float32),
        "camera_R": np.stack([c.R for c in cameras]).astype(np.float32),
        "camera_t": np.stack([c.t for c in cameras]).astype(np.float32),
        "camera_K": np.stack([c.K for c in cameras]).astype(np.float32),
        "camera_direction": np.stack([c.direction for c in cameras]).astype(np.float32),
        "camera_size": np.asarray([c.size for c in cameras], dtype=np.int32),
    }
    payload.update(extra_arrays)
    np.savez_compressed(path, **payload)
    return path
