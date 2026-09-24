"""Virtual-camera fracture detection for textured ContextCapture OBJ scenes.

The camera renders a continuous slope view, records per-pixel world XYZ from
the depth buffer, performs the existing fixed-window majority vote in image
space, and maps the resulting skeleton directly back to the visible surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from skimage.morphology import skeletonize

from tools import fracture_3d_analysis as base


PathLike = Union[str, Path]
CAMERA_SCHEMA_VERSION = 4
CAPTURE_ALGORITHM_VERSION = 4
INFERENCE_ALGORITHM_VERSION = 3
MEASUREMENT_ALGORITHM_VERSION = 2
VISUALIZATION_ALGORITHM_VERSION = 1
# Preserve parsed OBJ arrays across ``importlib.reload`` in the notebook. This
# makes recovery from an interrupted selector substantially faster.
_OBJ_MESH_CACHE: Dict[
    Tuple[str, int, int], Dict[str, np.ndarray]
] = globals().get('_OBJ_MESH_CACHE', {})


def _require_pyvista():
    try:
        import pyvista as pv
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'Virtual-camera capture requires PyVista and VTK. Install the '
            'compatible packages with: pip install pyvista==0.44.2 '
            'vtk==9.2.6') from error
    return pv


def _file_signature(path: PathLike) -> Dict:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return dict(
        path=str(path), size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns))


def _stable_fingerprint(value: Dict, length: int = 12) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:length]


def _directory_signature(directory: PathLike,
                         pattern: str = '*.py') -> Dict:
    """Return a cheap deterministic signature for local implementation code."""
    directory = Path(directory).expanduser().resolve()
    records = []
    if directory.is_dir():
        for path in sorted(directory.rglob(pattern)):
            if path.is_file():
                stat = path.stat()
                records.append((
                    str(path.relative_to(directory)).replace('\\', '/'),
                    int(stat.st_size), int(stat.st_mtime_ns)))
    return dict(path=str(directory), fingerprint=_stable_fingerprint(
        {'files': records}, length=20), file_count=len(records))


def _resolved_config_signature(path: PathLike) -> Dict:
    """Fingerprint the merged config, including its ``_base_`` files."""
    from mmengine.config import Config

    path = Path(path).expanduser().resolve()
    config = Config.fromfile(str(path))
    merged_text = config.pretty_text
    return dict(
        file=_file_signature(path),
        merged_sha1=hashlib.sha1(
            merged_text.encode('utf-8')).hexdigest())


def _dominant_winding_normal(triangles: np.ndarray) -> Optional[np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        return None
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(cross).all(axis=1) & (lengths > 1e-12)
    if not valid.any():
        return None
    normal = cross[valid].sum(axis=0)
    norm = np.linalg.norm(normal)
    return None if norm <= 1e-12 else normal / norm


def _triangle_centres_in_camera(triangles: np.ndarray, camera: Dict,
                                resolution: Tuple[int, int]) -> np.ndarray:
    centres = np.asarray(triangles, dtype=np.float64).mean(axis=1)
    position, focal, right, up, forward = _camera_basis(camera)
    relative = centres - focal
    x = relative @ right
    y = relative @ up
    depth = (centres - position) @ forward
    half_height = float(camera['parallel_scale'])
    half_width = half_height * resolution[0] / max(resolution[1], 1)
    return ((np.abs(x) <= half_width) & (np.abs(y) <= half_height)
            & (depth > 0))


def _frontmost_triangle_mask(triangles: np.ndarray, camera: Dict,
                             resolution: Tuple[int, int]) -> np.ndarray:
    """Approximate visible faces with a coarse orthographic z-buffer."""
    triangles = np.asarray(triangles, dtype=np.float64)
    inside = _triangle_centres_in_camera(triangles, camera, resolution)
    selected_indices = np.flatnonzero(inside)
    visible = np.zeros(len(triangles), dtype=bool)
    if not len(selected_indices):
        return visible
    centres = triangles[selected_indices].mean(axis=1)
    position, focal, right, up, forward = _camera_basis(camera)
    relative = centres - focal
    half_height = float(camera['parallel_scale'])
    aspect = resolution[0] / max(resolution[1], 1)
    half_width = half_height * aspect
    # A moderate grid is deliberately used here: the scene sample is sparse,
    # while keeping only the nearest centre in each ray bin removes fully
    # occluded bench/back faces from the local normal estimate.
    grid_width = min(max(int(resolution[0] // 4), 64), 512)
    grid_height = min(max(int(resolution[1] // 4), 64), 512)
    columns = np.clip(np.floor(
        ((relative @ right) + half_width) / (2 * half_width)
        * grid_width).astype(int), 0, grid_width - 1)
    rows = np.clip(np.floor(
        ((relative @ up) + half_height) / (2 * half_height)
        * grid_height).astype(int), 0, grid_height - 1)
    depth = (centres - position) @ forward
    bins = rows.astype(np.int64) * grid_width + columns
    order = np.lexsort((depth, bins))
    sorted_bins = bins[order]
    first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
    visible[selected_indices[order[first]]] = True
    return visible


def _fit_selected_surface_normal(
        sample: Dict, camera: Dict, resolution: Tuple[int, int],
        min_triangles: int = 30) -> Dict:
    """Align a camera to a robust mean normal of its selected mesh patch."""
    triangles = np.asarray(sample['triangles'], dtype=np.float64)
    inside = _frontmost_triangle_mask(triangles, camera, resolution)
    selected = triangles[inside]
    if len(selected) < min_triangles:
        fitted = dict(camera)
        fitted['local_normal_triangle_count'] = int(len(selected))
        fitted['local_normal_fitted'] = False
        return fitted
    cross = np.cross(
        selected[:, 1] - selected[:, 0],
        selected[:, 2] - selected[:, 0])
    areas2 = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(cross).all(axis=1) & (areas2 > 1e-12)
    cross, areas2 = cross[valid], areas2[valid]
    if len(cross) < min_triangles:
        fitted = dict(camera)
        fitted['local_normal_triangle_count'] = int(len(cross))
        fitted['local_normal_fitted'] = False
        return fitted
    normals = cross / areas2[:, None]
    position, focal, _, _, current_forward = _camera_basis(camera)
    toward_camera = -current_forward
    normals[np.einsum('ij,j->i', normals, toward_camera) < 0] *= -1
    weights = np.minimum(areas2, np.percentile(areas2, 90))
    mean = np.average(normals, axis=0, weights=weights)
    mean /= max(np.linalg.norm(mean), 1e-12)
    # Reject strongly discordant facets (vegetation, side faces), then refit.
    consistent = normals @ mean >= math.cos(math.radians(55))
    if int(consistent.sum()) >= min_triangles:
        mean = np.average(
            normals[consistent], axis=0, weights=weights[consistent])
        mean /= max(np.linalg.norm(mean), 1e-12)
        used = consistent
    else:
        used = np.ones(len(normals), dtype=bool)
    local_forward = -mean
    # Put the rotation centre on the selected surface. Moving both camera
    # points along the old optical axis does not change an orthographic view,
    # but prevents the framed patch from swinging away when its local normal
    # differs from the global best-fit plane.
    selected_centres = selected.mean(axis=1)[valid][used]
    surface_offset = float(np.median(
        (selected_centres - focal) @ current_forward))
    centred = dict(camera)
    axial_shift = current_forward * surface_offset
    centred['position'] = (position + axial_shift).tolist()
    centred['focal_point'] = (focal + axial_shift).tolist()
    old_up = _camera_basis(camera)[3]
    fitted = _snap_camera_to_normal(centred, local_forward, old_up)
    fitted.update(
        normal_source='selected_region_robust_mesh_normal',
        local_normal_fitted=True,
        local_normal_triangle_count=int(used.sum()),
        local_surface_axial_offset_m=surface_offset,
        local_normal_change_deg=float(math.degrees(math.acos(np.clip(
            np.dot(current_forward, local_forward), -1.0, 1.0)))))
    return fitted


def _camera_basis(camera: Dict) -> Tuple[np.ndarray, ...]:
    position = np.asarray(camera['position'], dtype=np.float64).copy()
    focal = np.asarray(camera['focal_point'], dtype=np.float64).copy()
    view_up = np.asarray(camera['view_up'], dtype=np.float64).copy()
    if (position.shape != (3,) or focal.shape != (3,)
            or view_up.shape != (3,)
            or not np.isfinite(np.concatenate(
                (position, focal, view_up))).all()):
        raise ValueError(
            'Camera position, focal point and view-up must be finite '
            '3D vectors.')
    forward = focal - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        raise ValueError('Camera position and focal point must differ.')
    forward /= forward_norm
    view_up -= forward * np.dot(view_up, forward)
    up_norm = np.linalg.norm(view_up)
    if up_norm <= 1e-12:
        raise ValueError(
            'Camera view-up cannot be parallel to its optical axis.')
    view_up /= up_norm
    right = np.cross(forward, view_up)
    right /= max(np.linalg.norm(right), 1e-12)
    return position, focal, right, view_up, forward


def _camera_state(plotter, resolution: Tuple[int, int],
                  job_keys: Optional[Sequence[str]] = None) -> Dict:
    camera = plotter.camera
    return dict(
        projection='orthographic',
        position=[float(value) for value in camera.position],
        focal_point=[float(value) for value in camera.focal_point],
        view_up=[float(value) for value in camera.up],
        parallel_scale=float(camera.parallel_scale),
        clipping_range=[float(value) for value in camera.clipping_range],
        resolution=[int(resolution[0]), int(resolution[1])],
        job_keys=list(job_keys or []))


def _apply_camera(plotter, camera: Dict) -> None:
    _camera_basis(camera)
    if (not np.isfinite(camera['parallel_scale'])
            or float(camera['parallel_scale']) <= 0):
        raise ValueError('Camera parallel_scale must be positive and finite.')
    # PyVista 0.44 recalculates ``parallel_scale`` when parallel projection is
    # enabled. Enable it first, then restore the exact selected scale last.
    plotter.enable_parallel_projection()
    plotter.camera.position = camera['position']
    plotter.camera.focal_point = camera['focal_point']
    plotter.camera.up = camera['view_up']
    plotter.camera.clipping_range = camera['clipping_range']
    plotter.camera.parallel_scale = float(camera['parallel_scale'])


def _sample_polydata(sample: Dict):
    pv = _require_pyvista()
    triangles = sample['triangles'].astype(np.float32, copy=False)
    points = triangles.reshape(-1, 3)
    face_indices = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
    faces = np.column_stack((
        np.full(len(face_indices), 3, dtype=np.int64), face_indices)).ravel()
    mesh = pv.PolyData(points, faces)
    mesh.cell_data['texture_rgb'] = sample['colors'].astype(np.uint8)
    return mesh


def _visible_sample_jobs(sample: Dict, camera: Dict,
                         resolution: Tuple[int, int]) -> List[str]:
    triangles = sample['triangles']
    position, focal, right, up, forward = _camera_basis(camera)
    relative = triangles - focal
    x = relative @ right
    y = relative @ up
    depth = (triangles - position) @ forward
    aspect = resolution[0] / max(resolution[1], 1)
    half_height = float(camera['parallel_scale']) * 1.05
    half_width = half_height * aspect
    # Test triangle bounds rather than only centroids; a large boundary face
    # can cross the viewport even when its centroid lies just outside it.
    inside = ((x.max(axis=1) >= -half_width)
              & (x.min(axis=1) <= half_width)
              & (y.max(axis=1) >= -half_height)
              & (y.min(axis=1) <= half_height)
              & (depth.max(axis=1) > 0))
    if not inside.any():
        return []
    job_indices = np.unique(sample['job_indices'][inside])
    job_keys = sample['metadata']['job_keys']
    return [job_keys[int(index)] for index in job_indices]


def _default_camera(scene_path: PathLike, cache_dir: PathLike,
                    resolution: Tuple[int, int],
                    sample: Optional[Dict] = None) -> Dict:
    overview = base.build_scene_overview(scene_path, cache_dir)
    center = np.asarray(overview['center'], dtype=np.float64)
    axes = np.asarray(overview['axes'], dtype=np.float64)
    lower, upper = np.asarray(overview['projection_bounds'], dtype=np.float64)
    width_m, height_m = upper - lower
    normal = axes[2].copy()
    view_up = axes[1].copy()
    if sample is not None and len(sample.get('triangles', [])):
        winding_normal = _dominant_winding_normal(sample['triangles'])
        if winding_normal is not None and np.dot(normal, winding_normal) < 0:
            normal *= -1
    forward = -normal
    right = np.cross(forward, view_up)
    right /= max(np.linalg.norm(right), 1e-12)
    if sample is not None and len(sample.get('triangles', [])):
        points = sample['triangles'].reshape(-1, 3).astype(np.float64)
        relative = points - center
        x_bounds = np.percentile(relative @ right, [0.1, 99.9])
        y_bounds = np.percentile(relative @ view_up, [0.1, 99.9])
        center += (x_bounds.mean() * right + y_bounds.mean() * view_up)
        width_m = float(np.diff(x_bounds)[0])
        height_m = float(np.diff(y_bounds)[0])
    position = center + normal * max(width_m, height_m) * 2.2
    return dict(
        projection='orthographic',
        position=position.tolist(),
        focal_point=center.tolist(),
        view_up=view_up.tolist(),
        parallel_scale=float(max(height_m / 2, width_m / 2 /
                                 (resolution[0] / resolution[1])) * 1.08),
        clipping_range=[0.1, float(max(width_m, height_m) * 6)],
        resolution=list(resolution),
        job_keys=[])


def _camera_from_viewport_rectangle(
        camera: Dict,
        viewport: Sequence[int],
        render_resolution: Tuple[int, int],
        capture_resolution: Tuple[int, int]) -> Optional[Dict]:
    """Convert a screen-space rectangle to a cropped orthographic camera."""
    if len(viewport) != 4:
        raise ValueError('viewport must contain x0, y0, x1 and y1.')
    render_width, render_height = map(int, render_resolution)
    capture_width, capture_height = map(int, capture_resolution)
    if (render_width <= 0 or render_height <= 0
            or capture_width <= 0 or capture_height <= 0):
        return None
    x0, y0, x1, y1 = map(float, viewport)
    left, right_pixel = sorted((x0, x1))
    bottom, top = sorted((y0, y1))
    if right_pixel - left < 8 or top - bottom < 8:
        return None
    left = float(np.clip(left, 0, render_width))
    right_pixel = float(np.clip(right_pixel, 0, render_width))
    bottom = float(np.clip(bottom, 0, render_height))
    top = float(np.clip(top, 0, render_height))
    if right_pixel - left < 8 or top - bottom < 8:
        return None

    position, focal, camera_right, camera_up, _ = _camera_basis(camera)
    half_height = float(camera['parallel_scale'])
    half_width = half_height * render_width / render_height
    center_x = ((left + right_pixel) / render_width - 1.0) * half_width
    center_y = ((bottom + top) / render_height - 1.0) * half_height
    selected_width = ((right_pixel - left) / render_width) * 2 * half_width
    selected_height = ((top - bottom) / render_height) * 2 * half_height
    capture_aspect = capture_width / capture_height
    new_parallel_scale = max(
        selected_height / 2, selected_width / (2 * capture_aspect))
    shift = center_x * camera_right + center_y * camera_up
    cropped = dict(camera)
    cropped.update(
        position=(position + shift).tolist(),
        focal_point=(focal + shift).tolist(),
        parallel_scale=float(new_parallel_scale),
        resolution=[capture_width, capture_height],
        selection_viewport=[int(left), int(bottom),
                            int(right_pixel), int(top)])
    return cropped


def _snap_camera_to_normal(camera: Dict, forward: np.ndarray,
                           view_up: np.ndarray) -> Dict:
    """Return a camera whose optical axis exactly follows a fixed normal."""
    snapped = dict(camera)
    position = np.asarray(camera['position'], dtype=np.float64)
    focal = np.asarray(camera['focal_point'], dtype=np.float64)
    distance = max(float(np.linalg.norm(focal - position)), 1e-6)
    forward = np.asarray(forward, dtype=np.float64).copy()
    if forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError('Locked camera normal must be a finite 3D vector.')
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        raise ValueError('Locked camera normal cannot be zero.')
    forward /= forward_norm
    view_up = np.asarray(view_up, dtype=np.float64)
    if view_up.shape != (3,) or not np.isfinite(view_up).all():
        raise ValueError('Locked camera view-up must be a finite 3D vector.')
    view_up -= forward * np.dot(view_up, forward)
    up_norm = np.linalg.norm(view_up)
    if up_norm <= 1e-12:
        raise ValueError(
            'Locked camera view-up cannot be parallel to its normal.')
    view_up /= up_norm
    snapped.update(
        position=(focal - forward * distance).tolist(),
        focal_point=focal.tolist(),
        view_up=view_up.tolist(),
        optical_axis=forward.tolist(),
        slope_plane_normal=(-forward).tolist(),
        normal_locked=True,
        normal_source='global_pca_best_fit_plane')
    return snapped


def select_virtual_camera(
        scene_path: PathLike,
        cache_dir: PathLike = 'fracture_3d_pipeline',
        capture_resolution: Tuple[int, int] = (3072, 1536),
        detection_window_size: Tuple[int, int] = (256, 256),
        preview_texture_dimension: int = 1024,
        lock_camera_normal: bool = True,
        max_projected_metres_per_pixel: Optional[float] = None,
        face_sample_step: int = 20,
        max_triangles: int = 200000) -> Optional[Dict]:
    """Open a textured model and frame a metric orthographic detection area."""
    pv = _require_pyvista()
    capture_resolution = tuple(map(int, capture_resolution))
    detection_window_size = tuple(map(int, detection_window_size))
    if (len(capture_resolution) != 2 or min(capture_resolution) < 16):
        raise ValueError('capture_resolution must contain two values >= 16.')
    if (len(detection_window_size) != 2
            or min(detection_window_size) < 1):
        raise ValueError(
            'detection_window_size must contain two positive values.')
    if preview_texture_dimension < 16:
        raise ValueError('preview_texture_dimension must be at least 16.')
    if (max_projected_metres_per_pixel is not None
            and max_projected_metres_per_pixel <= 0):
        raise ValueError(
            'max_projected_metres_per_pixel must be positive or None.')
    cache_dir = Path(cache_dir).expanduser().resolve()
    sample = base.build_scene_3d_sample(
        scene_path, cache_dir, face_sample_step, max_triangles)
    initial = _default_camera(
        scene_path, cache_dir, capture_resolution, sample=sample)
    _, _, _, locked_up, locked_forward = _camera_basis(initial)
    if lock_camera_normal:
        initial = _snap_camera_to_normal(
            initial, locked_forward, locked_up)
    state = dict(
        camera=None, pending_camera=None, done=False, cancelled=False)
    # Force a native VTK window. Inside Jupyter, PyVista otherwise tries the
    # optional trame backend and silently falls back to a non-interactive image
    # when trame is unavailable.
    capture_aspect = capture_resolution[0] / max(capture_resolution[1], 1)
    if capture_aspect >= 1:
        preview_resolution = (1280, max(480, round(1280 / capture_aspect)))
    else:
        preview_resolution = (max(480, round(800 * capture_aspect)), 800)
    plotter = pv.Plotter(
        notebook=False, window_size=preview_resolution, title=(
            'Textured virtual camera: R box-selects, C captures view'))
    plotter.set_background('#181818')
    jobs = base.discover_scene_texture_jobs(scene_path)
    # Load every texture job for the interactive preview. A sparse sample is
    # useful for camera fitting but is not safe as a hard visibility filter:
    # small boundary materials can otherwise disappear after a pan.
    preview_jobs = jobs
    preview_assets = []
    try:
        from tqdm import tqdm
        iterator = tqdm(
            preview_jobs, desc='Loading textured 3D preview', unit='tile')
    except ImportError:
        iterator = preview_jobs
    for job in iterator:
        for textured_mesh, texture in _textured_job_meshes(
                job, initial, preview_resolution,
                preview_texture_dimension):
            preview_assets.append((textured_mesh, texture))
            plotter.add_mesh(
                textured_mesh, texture=texture, lighting=False,
                show_edges=False)
    if not preview_assets:
        mesh = _sample_polydata(sample)
        plotter.add_mesh(
            mesh, scalars='texture_rgb', rgb=True, preference='cell',
            lighting=False, show_edges=False)
    navigation_text = (
        'Slope-normal view locked: middle drag pans, wheel zooms\n'
        if lock_camera_normal else
        'Left drag / wheel: rotate and zoom\n')
    instruction_actor = plotter.add_text(
        navigation_text
        + 'R: box mode, then drag a rectangle\n'
        + 'C: capture the whole current view    Q: cancel',
        position='upper_left', font_size=11, color='white')
    scale_actor = plotter.add_text(
        '', position='lower_left', font_size=10, color='white')
    _apply_camera(plotter, initial)

    def update_scale(*_):
        metres_per_pixel = (
            2 * float(plotter.camera.parallel_scale)
            / max(capture_resolution[1], 1))
        projected_width = detection_window_size[0] * metres_per_pixel
        projected_height = detection_window_size[1] * metres_per_pixel
        scale_actor.set_text(
            0, f'{metres_per_pixel:.4f} m/pixel    '
               f'window: {projected_width:.2f} x '
               f'{projected_height:.2f} m (camera plane)')

    plotter.camera.AddObserver('ModifiedEvent', update_scale)
    update_scale()

    def restore_locked_normal(*_):
        if not lock_camera_normal or state['done']:
            return
        current = _camera_state(plotter, capture_resolution)
        snapped = _snap_camera_to_normal(
            current, locked_forward, locked_up)
        _apply_camera(plotter, snapped)
        update_scale()

    plotter.iren.add_observer('EndInteractionEvent', restore_locked_normal)

    def persist_camera(camera):
        camera_dir = cache_dir / 'virtual_camera'
        camera_dir.mkdir(parents=True, exist_ok=True)
        base._atomic_save_json(camera_dir / 'last_camera.json', camera)

    def capture():
        current = _camera_state(plotter, capture_resolution)
        render_size = tuple(map(int, plotter.render_window.GetSize()))
        camera = _camera_from_viewport_rectangle(
            current, (0, 0, render_size[0], render_size[1]),
            render_size, capture_resolution)
        if camera is None:
            return
        final_normal_fitted = False
        if lock_camera_normal:
            camera = _snap_camera_to_normal(
                camera, locked_forward, locked_up)
            final_fit = _fit_selected_surface_normal(
                sample, camera, capture_resolution)
            if final_fit.get('local_normal_fitted', False):
                camera = final_fit
                final_normal_fitted = True
                _, _, _, local_up, local_forward = _camera_basis(camera)
                locked_forward[:] = local_forward
                locked_up[:] = local_up
        pending_camera = state.get('pending_camera')
        if pending_camera is not None:
            # The viewport is useful provenance. Normal metadata must describe
            # the final, possibly panned view fitted above, not the older box.
            if 'selection_viewport' in pending_camera:
                camera['selection_viewport'] = pending_camera[
                    'selection_viewport']
            if lock_camera_normal and not final_normal_fitted:
                for key in (
                        'local_normal_fitted',
                        'local_normal_triangle_count',
                        'local_normal_change_deg',
                        'local_surface_axial_offset_m', 'normal_source'):
                    if key in pending_camera:
                        camera[key] = pending_camera[key]
        camera['camera_schema_version'] = CAMERA_SCHEMA_VERSION
        camera['scene_path'] = str(Path(scene_path).expanduser().resolve())
        metres_per_pixel = (
            2 * float(camera['parallel_scale']) / capture_resolution[1])
        if (max_projected_metres_per_pixel is not None
                and metres_per_pixel > max_projected_metres_per_pixel):
            instruction_actor.set_text(
                2, 'Selected view is too coarse for configured detection\n'
                   f'{metres_per_pixel:.4f} m/pixel > '
                   f'{max_projected_metres_per_pixel:.4f}; zoom in, then C')
            return
        camera['metres_per_pixel'] = metres_per_pixel
        camera['window_projected_size_m'] = [
            float(detection_window_size[0] * metres_per_pixel),
            float(detection_window_size[1] * metres_per_pixel)]
        camera['job_keys'] = _visible_sample_jobs(
            sample, camera, capture_resolution)
        state['camera'] = camera
        state['done'] = True
        persist_camera(camera)
        print(
            f'Virtual-camera area confirmed: {len(camera["job_keys"])} '
            'sampled candidate texture jobs. Leaving selector...')

    def capture_rectangle(selection):
        current = _camera_state(plotter, capture_resolution)
        if lock_camera_normal:
            current = _snap_camera_to_normal(
                current, locked_forward, locked_up)
        render_size = tuple(map(int, plotter.render_window.GetSize()))
        camera = _camera_from_viewport_rectangle(
            current, selection.viewport, render_size, capture_resolution)
        if camera is None:
            return
        if lock_camera_normal:
            camera = _fit_selected_surface_normal(
                sample, camera, capture_resolution)
            _, _, _, local_up, local_forward = _camera_basis(camera)
            locked_forward[:] = local_forward
            locked_up[:] = local_up
        camera['camera_schema_version'] = CAMERA_SCHEMA_VERSION
        metres_per_pixel = (
            2 * float(camera['parallel_scale']) / capture_resolution[1])
        camera['metres_per_pixel'] = metres_per_pixel
        camera['window_projected_size_m'] = [
            float(detection_window_size[0] * metres_per_pixel),
            float(detection_window_size[1] * metres_per_pixel)]
        camera['job_keys'] = _visible_sample_jobs(
            sample, camera, capture_resolution)
        state['pending_camera'] = dict(camera)
        # Closing a native VTK window from inside its rectangle-pick observer
        # can stall the interactor. Instead, frame the selected rectangle and
        # let the user confirm it with C from the normal key-event path.
        _apply_camera(plotter, camera)
        instruction_actor.set_text(
            2, 'Rectangle selected, locally normal-aligned and framed\n'
               'Middle drag pans without changing scale; wheel zooms\n'
               'Press C to confirm, or R and drag to replace it')
        update_scale()
        plotter.render()

    def cancel():
        state['camera'] = None
        state['pending_camera'] = None
        state['cancelled'] = True
        state['done'] = True

    def window_closed(*_):
        if not state['done']:
            state['cancelled'] = True
            state['done'] = True

    for key in ('c', 'C'):
        plotter.clear_events_for_key(key)
        plotter.add_key_event(key, capture)
    for key in ('q', 'Q'):
        plotter.clear_events_for_key(key)
        plotter.add_key_event(key, cancel)
    plotter.iren.add_observer('ExitEvent', window_closed)
    plotter.enable_rectangle_picking(
        callback=capture_rectangle,
        show_message=False,
        start=False,
        show_frustum=False)
    plotter.show(auto_close=False, interactive_update=True)
    try:
        while not state['done']:
            try:
                # ``Plotter.update`` renders unconditionally after processing
                # events. On Windows the close button may already have
                # destroyed the render window at that point. Process events
                # directly and let VTK interaction callbacks request redraws.
                plotter.iren.process_events()
            except (AttributeError, RuntimeError):
                # Closing a native Windows render window can destroy the VTK
                # object before its ExitEvent reaches Python.
                window_closed()
            time.sleep(0.01)
    finally:
        plotter.close()
    if state['cancelled'] or state['camera'] is None:
        return None
    persist_camera(state['camera'])
    return state['camera']


def load_last_virtual_camera(
        cache_dir: PathLike = 'fracture_3d_pipeline') -> Optional[Dict]:
    path = (Path(cache_dir).expanduser().resolve()
            / 'virtual_camera' / 'last_camera.json')
    if not path.is_file():
        return None
    try:
        camera = json.loads(path.read_text(encoding='utf-8'))
        _camera_basis(camera)
        scale = float(camera['parallel_scale'])
        resolution = tuple(map(int, camera['resolution']))
        if (not np.isfinite(scale) or scale <= 0 or len(resolution) != 2
                or min(resolution) < 16):
            return None
        return camera
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _faces_in_camera(triangle_xyz: np.ndarray, camera: Dict,
                     resolution: Tuple[int, int]) -> np.ndarray:
    position, focal, right, up, forward = _camera_basis(camera)
    relative = triangle_xyz - focal
    x = relative @ right
    y = relative @ up
    depth = (triangle_xyz - position) @ forward
    aspect = resolution[0] / max(resolution[1], 1)
    half_height = float(camera['parallel_scale']) * 1.08
    half_width = half_height * aspect
    # Keep any triangle intersecting the viewport. Centre-only filtering can
    # remove large boundary triangles and leave black wedges in the render.
    return ((x.max(axis=1) >= -half_width)
            & (x.min(axis=1) <= half_width)
            & (y.max(axis=1) >= -half_height)
            & (y.min(axis=1) <= half_height)
            & (depth.max(axis=1) > 0))


def _load_textured_obj_cached(obj_path: PathLike) -> Dict[str, np.ndarray]:
    path = Path(obj_path).expanduser().resolve()
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    mesh = _OBJ_MESH_CACHE.get(key)
    if mesh is None:
        mesh = base.load_textured_obj(path)
        _OBJ_MESH_CACHE[key] = mesh
    return mesh


def _read_texture_reduced(path: PathLike,
                          max_dimension: int) -> Optional[np.ndarray]:
    path = Path(path)
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
    except (ImportError, OSError):
        width = height = max_dimension
    ratio = max(width, height) / max(max_dimension, 1)
    if ratio >= 8:
        flag = cv2.IMREAD_REDUCED_COLOR_8
    elif ratio >= 4:
        flag = cv2.IMREAD_REDUCED_COLOR_4
    elif ratio >= 2:
        flag = cv2.IMREAD_REDUCED_COLOR_2
    else:
        flag = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        return None
    image_height, image_width = image.shape[:2]
    scale = min(1.0, max_dimension / max(image_height, image_width))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, round(image_width * scale)),
             max(1, round(image_height * scale))),
            interpolation=cv2.INTER_AREA)
    return image


def _textured_job_meshes(job: Dict, camera: Dict,
                         resolution: Tuple[int, int],
                         max_texture_dimension: int):
    pv = _require_pyvista()
    polydata_records = []
    mesh = _load_textured_obj_cached(job['obj_path'])
    for material in job.get('materials', []):
        material_faces = mesh['face_materials'] == material
        if not material_faces.any():
            continue
        vertex_faces = mesh['vertex_faces'][material_faces]
        uv_faces = mesh['uv_faces'][material_faces]
        triangle_xyz = mesh['vertices'][vertex_faces]
        keep = _faces_in_camera(triangle_xyz, camera, resolution)
        if not keep.any():
            continue
        triangle_xyz = triangle_xyz[keep].astype(np.float32)
        triangle_uv = mesh['texcoords'][uv_faces[keep]].astype(
            np.float32)
        points = triangle_xyz.reshape(-1, 3)
        texture_coordinates = triangle_uv.reshape(-1, 2)
        indices = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
        faces = np.column_stack((
            np.full(len(indices), 3, dtype=np.int64), indices)).ravel()
        polydata = pv.PolyData(points, faces)
        polydata.active_texture_coordinates = texture_coordinates
        polydata_records.append(polydata)
    if not polydata_records:
        return []
    texture_bgr = _read_texture_reduced(
        job['texture_path'], max_texture_dimension)
    if texture_bgr is None:
        return []
    texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
    # PyVista's NumPy texture bridge already accounts for the image-row/OBJ-v
    # convention. An extra vertical flip would invert the aerial texture.
    texture = pv.numpy_to_texture(texture_rgb)
    return [(polydata, texture) for polydata in polydata_records]


def _depth_to_world(depth: np.ndarray, camera: Dict) -> np.ndarray:
    height, width = depth.shape
    position, _, right, up, forward = _camera_basis(camera)
    half_height = float(camera['parallel_scale'])
    half_width = half_height * width / max(height, 1)
    x = np.linspace(-half_width + half_width / width,
                    half_width - half_width / width, width)
    y = np.linspace(half_height - half_height / height,
                    -half_height + half_height / height, height)
    world = np.full((height, width, 3), np.nan, dtype=np.float32)
    valid = np.isfinite(depth)
    rows, columns = np.nonzero(valid)
    if len(rows):
        distances = -depth[rows, columns].astype(np.float64)
        xyz = (position
               + x[columns, None] * right
               + y[rows, None] * up
               + distances[:, None] * forward)
        world[rows, columns] = xyz.astype(np.float32)
    return world


def _orthographic_depth_buffer(plotter,
                               reset_clipping: bool = True) -> np.ndarray:
    """Read VTK z-buffer and convert it to signed camera distance.

    PyVista 0.44's parallel-projection conversion treats the OpenGL z-buffer
    as if it were already a metric depth and can mark the whole image null.
    For an orthographic camera the correct mapping is linear between clipping
    planes.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

    if reset_clipping:
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()
    image_filter = vtkWindowToImageFilter()
    image_filter.SetInput(plotter.render_window)
    image_filter.ReadFrontBufferOff()
    image_filter.SetInputBufferTypeToZBuffer()
    image_filter.Update()
    output = image_filter.GetOutput()
    width, height, _ = output.GetDimensions()
    z_buffer = vtk_to_numpy(
        output.GetPointData().GetScalars()).reshape(height, width)
    z_buffer = np.flipud(z_buffer).astype(np.float32)
    near, far = plotter.camera.clipping_range
    depth = -(near + z_buffer * (far - near)).astype(np.float32)
    depth[z_buffer >= 1 - 1e-7] = np.nan
    return depth


def capture_virtual_camera_view(
        scene_path: PathLike,
        camera: Dict,
        output_dir: PathLike = 'fracture_3d_pipeline',
        max_texture_dimension: int = 4096,
        overwrite: bool = False) -> Dict:
    """Render camera RGB/depth/XYZ products from full textured OBJ faces."""
    pv = _require_pyvista()
    output_dir = Path(output_dir).expanduser().resolve()
    resolution = tuple(map(int, camera.get('resolution', (3072, 1536))))
    jobs = sorted(
        base.discover_scene_texture_jobs(scene_path),
        key=lambda job: job['cache_key'])
    requested = set(camera.get('job_keys') or [])
    # Sparse sample hits are only a loading priority, never a hard filter.
    # Every full OBJ is checked against the final camera to avoid black holes
    # when a small material was absent from the sampled preview.
    render_jobs = sorted(
        jobs, key=lambda job: (job['cache_key'] not in requested,
                               job['cache_key']))
    source_assets = []
    seen_obj_paths = set()
    for job in jobs:
        obj_path = str(Path(job['obj_path']).resolve())
        if obj_path not in seen_obj_paths:
            source_assets.append(_file_signature(obj_path))
            mtl_path = Path(obj_path).with_suffix('.mtl')
            if mtl_path.is_file():
                source_assets.append(_file_signature(mtl_path))
            seen_obj_paths.add(obj_path)
        source_assets.append(_file_signature(job['texture_path']))
    fingerprint_value = dict(
        capture_algorithm_version=CAPTURE_ALGORITHM_VERSION,
        camera={key: camera[key] for key in (
            'position', 'focal_point', 'view_up', 'parallel_scale')},
        resolution=resolution,
        scene_path=str(Path(scene_path).expanduser().resolve()),
        job_keys=[job['cache_key'] for job in jobs],
        max_texture_dimension=int(max_texture_dimension),
        renderer=dict(
            pyvista_version=str(getattr(pv, '__version__', 'unknown')),
            vtk_version='.'.join(map(str, pv.vtk_version_info))),
        source_assets=source_assets)
    fingerprint = _stable_fingerprint(fingerprint_value)
    capture_dir = output_dir / 'virtual_camera' / f'capture_{fingerprint}'
    capture_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = capture_dir / 'camera_rgb.png'
    depth_path = capture_dir / 'camera_depth.npy'
    xyz_path = capture_dir / 'camera_world_xyz.npz'
    camera_path = capture_dir / 'camera.json'
    metres_per_pixel = float(
        2 * camera['parallel_scale'] / resolution[1])
    if (not overwrite and rgb_path.is_file() and depth_path.is_file()
            and xyz_path.is_file() and camera_path.is_file()):
        try:
            saved_camera = json.loads(camera_path.read_text(encoding='utf-8'))
            cached_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            cached_depth = np.load(depth_path, mmap_mode='r')
            with np.load(xyz_path) as xyz_archive:
                cached_xyz_shape = xyz_archive['xyz'].shape
            valid_cache = (
                saved_camera.get('capture_fingerprint') == fingerprint
                and saved_camera.get('capture_algorithm_version')
                == CAPTURE_ALGORITHM_VERSION
                and cached_rgb is not None
                and cached_rgb.shape[:2] == (resolution[1], resolution[0])
                and cached_depth.shape == cached_rgb.shape[:2]
                and cached_xyz_shape == (*cached_rgb.shape[:2], 3))
            del cached_depth
            if valid_cache:
                rendered_job_keys = saved_camera.get(
                    'rendered_job_keys', [])
                return dict(
                    directory=str(capture_dir), rgb=str(rgb_path),
                    depth=str(depth_path), xyz=str(xyz_path),
                    camera=str(camera_path), fingerprint=fingerprint,
                    job_count=len(rendered_job_keys),
                    rendered_job_keys=rendered_job_keys,
                    metres_per_pixel=metres_per_pixel)
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            pass

    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.set_background('black')
    retained = []
    rendered_job_keys = []
    try:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                render_jobs, desc='Rendering selected textured area',
                unit='tile')
        except ImportError:
            iterator = render_jobs
        for job in iterator:
            job_meshes = _textured_job_meshes(
                job, camera, resolution, max_texture_dimension)
            if job_meshes:
                rendered_job_keys.append(job['cache_key'])
            for mesh, texture in job_meshes:
                retained.append((mesh, texture))
                plotter.add_mesh(
                    mesh, texture=texture, lighting=False, show_edges=False)
        if not retained:
            raise ValueError(
                'The virtual camera viewport contains no OBJ faces.')
        _apply_camera(plotter, camera)
        # Use one clipping state for both color and depth so their silhouettes
        # are pixel-aligned at near/far boundaries.
        plotter.renderer.ResetCameraClippingRange()
        # ``screenshot`` performs the first render. Avoid ``show`` here because
        # PyVista 0.44 also reads an unused full-resolution color/depth pair.
        rgb = plotter.screenshot(return_img=True)
        depth = _orthographic_depth_buffer(
            plotter, reset_clipping=False)
        render_clipping_range = [
            float(value) for value in plotter.camera.clipping_range]
    finally:
        plotter.close()
    if rgb.shape[:2] != depth.shape:
        raise RuntimeError(
            f'RGB/depth render mismatch: {rgb.shape[:2]} versus '
            f'{depth.shape}.')
    world_xyz = _depth_to_world(depth, camera)
    temporary_rgb = rgb_path.with_name(f'{rgb_path.stem}.tmp{rgb_path.suffix}')
    if not cv2.imwrite(
            str(temporary_rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not save virtual-camera RGB image: {rgb_path}')
    temporary_rgb.replace(rgb_path)
    base._atomic_save_npy(depth_path, depth.astype(np.float32))
    temporary_xyz = xyz_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary_xyz, xyz=world_xyz)
    temporary_xyz.replace(xyz_path)
    saved_camera = dict(camera)
    saved_camera.update(
        resolution=list(resolution),
        clipping_range=render_clipping_range,
        metres_per_pixel=metres_per_pixel,
        capture_fingerprint=fingerprint,
        capture_algorithm_version=CAPTURE_ALGORITHM_VERSION,
        rendered_job_keys=rendered_job_keys)
    base._atomic_save_json(camera_path, saved_camera)
    print(
        f'Virtual-camera RGB/depth/XYZ saved: {capture_dir} '
        f'({metres_per_pixel:.5f} m/pixel)')
    return dict(
        directory=str(capture_dir), rgb=str(rgb_path),
        depth=str(depth_path), xyz=str(xyz_path),
        camera=str(camera_path), fingerprint=fingerprint,
        job_count=len(rendered_job_keys),
        rendered_job_keys=rendered_job_keys,
        metres_per_pixel=metres_per_pixel)


def _typical_world_pixel_step(world_xyz: np.ndarray) -> float:
    distances = []
    valid = np.isfinite(world_xyz).all(axis=2)
    for row_step, col_step in ((0, 1), (1, 0)):
        source_valid = valid[:valid.shape[0] - row_step or None,
                             :valid.shape[1] - col_step or None]
        target_valid = valid[row_step:, col_step:]
        pairs = source_valid & target_valid
        if pairs.any():
            source = world_xyz[:world_xyz.shape[0] - row_step or None,
                               :world_xyz.shape[1] - col_step or None]
            target = world_xyz[row_step:, col_step:]
            values = np.linalg.norm(target[pairs] - source[pairs], axis=1)
            values = values[np.isfinite(values) & (values > 0)]
            if len(values) > 200000:
                values = values[np.linspace(
                    0, len(values) - 1, 200000).astype(int)]
            distances.append(values)
    if not distances:
        return float('nan')
    values = np.concatenate(distances)
    return float(np.median(values)) if len(values) else float('nan')


def _ordered_skeleton_paths(skeleton: np.ndarray,
                            world_xyz: np.ndarray,
                            depth_jump_factor: float = 8.0
                            ) -> List[List[Tuple[int, int]]]:
    """Build ordered, depth-aware paths and split them at junctions."""
    skeleton = np.asarray(skeleton, dtype=bool)
    world_xyz = np.asarray(world_xyz)
    if world_xyz.shape != (*skeleton.shape, 3):
        raise ValueError(
            'world_xyz must have shape skeleton.shape + (3,).')
    if depth_jump_factor <= 0:
        raise ValueError('depth_jump_factor must be positive.')
    coordinates = np.argwhere(skeleton)
    if not len(coordinates):
        return []
    node_ids = np.full(skeleton.shape, -1, dtype=np.int32)
    node_ids[coordinates[:, 0], coordinates[:, 1]] = np.arange(
        len(coordinates), dtype=np.int32)
    typical_step = _typical_world_pixel_step(world_xyz)
    max_normalized_step = (
        typical_step * depth_jump_factor
        if np.isfinite(typical_step) and typical_step > 0 else float('inf'))
    adjacency: List[List[int]] = [[] for _ in range(len(coordinates))]
    for row_step, col_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if col_step >= 0:
            source = skeleton[:skeleton.shape[0] - row_step or None,
                              :skeleton.shape[1] - col_step or None]
            target = skeleton[row_step:, col_step:]
            source_ids = node_ids[:node_ids.shape[0] - row_step or None,
                                  :node_ids.shape[1] - col_step or None]
            target_ids = node_ids[row_step:, col_step:]
            source_xyz = world_xyz[:world_xyz.shape[0] - row_step or None,
                                   :world_xyz.shape[1] - col_step or None]
            target_xyz = world_xyz[row_step:, col_step:]
        else:
            source = skeleton[:skeleton.shape[0] - row_step or None, 1:]
            target = skeleton[row_step:, :-1]
            source_ids = node_ids[:node_ids.shape[0] - row_step or None, 1:]
            target_ids = node_ids[row_step:, :-1]
            source_xyz = world_xyz[
                :world_xyz.shape[0] - row_step or None, 1:]
            target_xyz = world_xyz[row_step:, :-1]
        pairs = source & target
        if row_step and col_step:
            # Do not add the diagonal shortcut around an existing L corner.
            rows, columns = np.nonzero(pairs)
            source_columns = columns + (1 if col_step < 0 else 0)
            target_columns = source_columns + col_step
            connector_a = skeleton[rows, target_columns]
            connector_b = skeleton[rows + row_step, source_columns]
            keep_corner = ~(connector_a | connector_b)
            pairs[:] = False
            if keep_corner.any():
                if col_step >= 0:
                    pairs[rows[keep_corner], columns[keep_corner]] = True
                else:
                    pairs[rows[keep_corner], columns[keep_corner]] = True
        if not pairs.any():
            continue
        source_edge_ids = source_ids[pairs]
        target_edge_ids = target_ids[pairs]
        edge_distances = np.linalg.norm(
            target_xyz[pairs] - source_xyz[pairs], axis=1)
        pixel_step = math.sqrt(row_step * row_step + col_step * col_step)
        keep = (np.isfinite(edge_distances)
                & (edge_distances / pixel_step <= max_normalized_step))
        for source_id, target_id in zip(
                source_edge_ids[keep], target_edge_ids[keep]):
            source_id, target_id = int(source_id), int(target_id)
            adjacency[source_id].append(target_id)
            adjacency[target_id].append(source_id)

    visited_edges = set()
    paths: List[List[int]] = []

    def walk(start: int, neighbour: int) -> List[int]:
        path = [start]
        previous, current = start, neighbour
        visited_edges.add((min(previous, current), max(previous, current)))
        while True:
            path.append(current)
            candidates = [value for value in adjacency[current]
                          if value != previous]
            if len(adjacency[current]) != 2 or not candidates:
                break
            next_node = candidates[0]
            edge = (min(current, next_node), max(current, next_node))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            previous, current = current, next_node
        return path

    for node, neighbours in enumerate(adjacency):
        if len(neighbours) == 2:
            continue
        if not neighbours:
            paths.append([node])
        for neighbour in neighbours:
            edge = (min(node, neighbour), max(node, neighbour))
            if edge not in visited_edges:
                paths.append(walk(node, neighbour))
    for node, neighbours in enumerate(adjacency):
        for neighbour in neighbours:
            edge = (min(node, neighbour), max(node, neighbour))
            if edge not in visited_edges:
                paths.append(walk(node, neighbour))
    return [[tuple(map(int, coordinates[node])) for node in path]
            for path in paths]


def _simplify_polyline(points: np.ndarray, tolerance: float,
                       max_points: int = 2000) -> np.ndarray:
    """Simplify an ordered 3D polyline without turning it into sparse dots."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        segment_length2 = float(np.dot(segment, segment))
        interior = points[start + 1:end]
        if segment_length2 <= 1e-20:
            distances = np.linalg.norm(interior - points[start], axis=1)
        else:
            fractions = np.clip(
                ((interior - points[start]) @ segment) / segment_length2,
                0.0, 1.0)
            closest = points[start] + fractions[:, None] * segment
            distances = np.linalg.norm(interior - closest, axis=1)
        local_index = int(np.argmax(distances))
        if distances[local_index] > tolerance:
            index = start + 1 + local_index
            keep[index] = True
            stack.extend(((start, index), (index, end)))
    simplified = points[keep]
    if len(simplified) > max_points:
        indices = np.unique(np.linspace(
            0, len(simplified) - 1, max_points).astype(int))
        simplified = simplified[indices]
    return simplified


def _measure_camera_prediction(
        prediction: np.ndarray,
        world_xyz: np.ndarray,
        min_component_pixels: int,
        min_trace_length_m: float
        ) -> Tuple[List[Dict], List[Dict], np.ndarray]:
    prediction = np.asarray(prediction)
    world_xyz = np.asarray(world_xyz)
    if prediction.ndim != 2 or world_xyz.shape != (*prediction.shape, 3):
        raise ValueError(
            'prediction must be 2D and world_xyz must have matching HxWx3.')
    valid_surface = np.isfinite(world_xyz).all(axis=2)
    skeleton = skeletonize((prediction > 0) & valid_surface)
    records = []
    geometry = []
    accepted_skeleton = np.zeros_like(skeleton, dtype=bool)
    paths = _ordered_skeleton_paths(skeleton, world_xyz)
    accepted_paths = []
    typical_step = _typical_world_pixel_step(world_xyz)
    simplification_tolerance = (
        max(float(typical_step) * 0.25, 1e-5)
        if np.isfinite(typical_step) else 1e-5)
    for path in paths:
        pixel_count = len(path)
        if pixel_count < min_component_pixels:
            continue
        rows = np.asarray([value[0] for value in path], dtype=int)
        columns = np.asarray([value[1] for value in path], dtype=int)
        points = world_xyz[rows, columns].astype(np.float64)
        length_m = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if not np.isfinite(length_m) or length_m < min_trace_length_m:
            continue
        segment_id = len(records) + 1
        centroid = points.mean(axis=0)
        centered = points - centroid
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        horizontal = math.hypot(direction[0], direction[1])
        strike = math.degrees(math.atan2(direction[0], direction[1])) % 180
        inclination = math.degrees(
            math.atan2(abs(direction[2]), horizontal))
        visual_points = _simplify_polyline(
            points, simplification_tolerance, max_points=2000)
        sampled_points = np.round(visual_points, 5).tolist()
        records.append(dict(
            tile_name='virtual_camera',
            texture_name='camera_rgb.png',
            component_id=0,
            segment_id=segment_id,
            skeleton_pixels=pixel_count,
            length_3d_m=float(length_m),
            centroid_e=float(centroid[0]),
            centroid_n=float(centroid[1]),
            centroid_z=float(centroid[2]),
            strike_deg=float(strike),
            trace_inclination_deg=float(inclination)))
        geometry.append(dict(
            component_id=0,
            segment_id=segment_id,
            points=sampled_points,
            length_3d_m=float(length_m)))
        accepted_paths.append(path)
        accepted_skeleton[rows, columns] = True
    # Paths are split at junctions for unambiguous length/strike measurements.
    # Give segments that share a junction the same fracture-network ID so a T
    # shape is one fracture/network with three trace segments, not three
    # separate cracks.
    parent = list(range(len(accepted_paths)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners = {}
    for path_index, path in enumerate(accepted_paths):
        for pixel in path:
            previous = owners.setdefault(pixel, path_index)
            union(path_index, previous)
    root_to_fracture = {}
    for index, (record, trace) in enumerate(zip(records, geometry)):
        root = find(index)
        fracture_id = root_to_fracture.setdefault(
            root, len(root_to_fracture) + 1)
        record['component_id'] = fracture_id
        trace['component_id'] = fracture_id
    return records, geometry, accepted_skeleton


def _validate_skeleton_display_width(display_width_px: int) -> int:
    if (isinstance(display_width_px, (bool, np.bool_))
            or not isinstance(display_width_px, (int, np.integer))):
        raise ValueError('skeleton_display_width_px must be an odd integer.')
    display_width_px = int(display_width_px)
    if display_width_px < 1 or display_width_px % 2 == 0:
        raise ValueError(
            'skeleton_display_width_px must be a positive odd integer.')
    return display_width_px


def _thicken_skeleton(accepted_skeleton: np.ndarray,
                      display_width_px: int) -> np.ndarray:
    """Return a display-only dilation without altering the metric skeleton."""
    display_width_px = _validate_skeleton_display_width(display_width_px)
    skeleton = np.asarray(accepted_skeleton, dtype=bool)
    if skeleton.ndim != 2:
        raise ValueError('accepted_skeleton must be a two-dimensional mask.')
    if display_width_px == 1:
        return skeleton.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (display_width_px, display_width_px))
    return cv2.dilate(skeleton.astype(np.uint8), kernel) > 0


def _save_camera_overlay(rgb: np.ndarray, accepted_skeleton: np.ndarray,
                         output_path: Path,
                         display_width_px: int = 9) -> Path:
    skeleton = np.asarray(accepted_skeleton, dtype=bool).copy()
    # Thicken only a display copy. Length, orientation, and 3D coordinates
    # continue to use the one-pixel cyan centreline.
    trace_band = _thicken_skeleton(skeleton, display_width_px)
    overlay = rgb.astype(np.float32)
    overlay[trace_band] = (
        overlay[trace_band] * 0.20
        + np.array([255, 20, 20], dtype=np.float32) * 0.80)
    overlay[skeleton] = np.array([0, 255, 255], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    if not cv2.imwrite(
            str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not save accepted trace overlay: {output_path}')
    return output_path


def _save_raw_prediction_overlay(rgb: np.ndarray, prediction: np.ndarray,
                                 surface_mask: np.ndarray,
                                 output_path: Path) -> Path:
    positive = (prediction > 0) & surface_mask.astype(bool)
    overlay = rgb.astype(np.float32)
    overlay[positive] = (
        overlay[positive] * 0.35
        + np.array([255, 30, 30], dtype=np.float32) * 0.65)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    if not cv2.imwrite(
            str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not save raw prediction overlay: {output_path}')
    return output_path


def save_fracture_statistical_plots(fractures,
                                    output_dir: PathLike,
                                    stem: str = 'fracture_3d',
                                    fracture_count: Optional[int] = None
                                    ) -> Dict[str, str]:
    """Save trace statistics and a bidirectional strike rose diagram."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    required_columns = {
        'length_3d_m', 'strike_deg', 'trace_inclination_deg'}
    missing_columns = required_columns.difference(fractures.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(f'Fracture table is missing columns: {missing}')

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / f'{stem}_statistics.png'
    rose_path = output_dir / f'{stem}_strike_rose.png'
    segment_count = len(fractures)
    if fracture_count is None:
        fracture_count = (fractures['component_id'].nunique()
                          if 'component_id' in fractures.columns else
                          segment_count)

    lengths = fractures['length_3d_m'].to_numpy(dtype=float)
    inclinations = fractures[
        'trace_inclination_deg'].to_numpy(dtype=float)
    strikes = fractures['strike_deg'].to_numpy(dtype=float) % 180
    finite = (np.isfinite(lengths) & np.isfinite(inclinations)
              & np.isfinite(strikes))
    lengths = lengths[finite]
    inclinations = inclinations[finite]
    strikes = strikes[finite]

    statistics = Figure(figsize=(13, 4.8), constrained_layout=True)
    FigureCanvasAgg(statistics)
    length_axis, inclination_axis, summary_axis = statistics.subplots(1, 3)
    statistics.suptitle(
        '3D fracture trace statistics', fontsize=15, fontweight='bold')
    if len(lengths):
        length_axis.hist(
            lengths, bins='auto', color='#2878B5', edgecolor='white',
            alpha=0.9)
        length_axis.axvline(
            float(np.mean(lengths)), color='#C82423', linestyle='--',
            linewidth=1.6, label=f'Mean: {np.mean(lengths):.2f} m')
        length_axis.legend(frameon=False, fontsize=9)
        inclination_axis.hist(
            inclinations, bins=np.linspace(0, 90, 10), color='#54A24B',
            edgecolor='white', alpha=0.9)
    else:
        for axis in (length_axis, inclination_axis):
            axis.text(
                0.5, 0.5, 'No traces passed the filters', ha='center',
                va='center', transform=axis.transAxes)
    length_axis.set(
        title='3D trace length distribution', xlabel='Length (m)',
        ylabel='Trace segment count')
    inclination_axis.set(
        title='Trace inclination distribution', xlabel='Inclination (deg)',
        ylabel='Trace segment count', xlim=(0, 90))
    for axis in (length_axis, inclination_axis):
        axis.grid(axis='y', alpha=0.2)

    summary_axis.axis('off')
    if len(lengths):
        summary_lines = [
            f'Fracture networks: {int(fracture_count)}',
            f'Trace segments: {segment_count}',
            f'Total trace length: {np.sum(lengths):.2f} m',
            f'Mean length: {np.mean(lengths):.2f} m',
            f'Median length: {np.median(lengths):.2f} m',
            f'Min / max length: {np.min(lengths):.2f} / '
            f'{np.max(lengths):.2f} m',
            f'Mean inclination: {np.mean(inclinations):.1f} deg',
            f'Median inclination: {np.median(inclinations):.1f} deg']
    else:
        summary_lines = [
            f'Fracture networks: {int(fracture_count)}',
            f'Trace segments: {segment_count}',
            'No finite measurements available']
    summary_axis.set_title('Summary', pad=12)
    summary_axis.text(
        0.04, 0.95, '\n'.join(summary_lines), ha='left', va='top',
        fontsize=11, linespacing=1.55, transform=summary_axis.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F4F6F7',
                  edgecolor='#B0B6BA'))
    statistics.savefig(statistics_path, dpi=200, bbox_inches='tight')
    statistics.clear()

    rose = Figure(figsize=(7.2, 7.2), constrained_layout=True)
    FigureCanvasAgg(rose)
    rose_axis = rose.add_subplot(1, 1, 1, projection='polar')
    if len(strikes):
        strike_radians = np.deg2rad(strikes)
        bins = np.linspace(0, np.pi, 19)
        counts, edges = np.histogram(strike_radians, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        rose_axis.bar(
            np.concatenate((centers, centers + np.pi)),
            np.concatenate((counts, counts)),
            width=np.concatenate((widths, widths)),
            color='#E6862A', edgecolor='white', linewidth=0.8, alpha=0.9)
    else:
        rose_axis.text(
            0.5, 0.5, 'No traces passed the filters', ha='center',
            va='center', transform=rose_axis.transAxes)
    rose_axis.set_theta_zero_location('N')
    rose_axis.set_theta_direction(-1)
    rose_axis.set_thetagrids(
        np.arange(0, 360, 30),
        labels=['N', '30°', '60°', 'E', '120°', '150°',
                'S', '210°', '240°', 'W', '300°', '330°'])
    rose_axis.set_title(
        '3D fracture strike rose diagram\n'
        '10° bins; radial scale = trace segment count',
        fontsize=14, fontweight='bold', pad=22)
    rose_axis.grid(alpha=0.35)
    rose.savefig(rose_path, dpi=200, bbox_inches='tight')
    rose.clear()
    return dict(
        fracture_statistics=str(statistics_path),
        strike_rose=str(rose_path))


def _show_virtual_camera_result(scene_path: PathLike, cache_dir: PathLike,
                                geometry: List[Dict], camera: Dict,
                                output_path: Path,
                                texture_dimension: int = 1024,
                                interactive: bool = True,
                                whole_scene_context: bool = True,
                                skeleton_display_width_px: int = 9) -> Path:
    skeleton_display_width_px = _validate_skeleton_display_width(
        skeleton_display_width_px)
    pv = _require_pyvista()
    capture_resolution = tuple(map(
        int, camera.get('resolution', (3072, 1536))))
    aspect = capture_resolution[0] / max(capture_resolution[1], 1)
    if aspect >= 1:
        viewer_resolution = (1600, max(480, round(1600 / aspect)))
    else:
        viewer_resolution = (max(480, round(1000 * aspect)), 1000)
    plotter = pv.Plotter(
        notebook=False, off_screen=not interactive,
        window_size=viewer_resolution,
        title='Virtual-camera fracture result (red)')
    plotter.set_background('#181818')
    sample = base.build_scene_3d_sample(scene_path, cache_dir)
    if whole_scene_context:
        view_camera = _default_camera(
            scene_path, cache_dir, viewer_resolution, sample=sample)
        jobs = sorted(
            base.discover_scene_texture_jobs(scene_path),
            key=lambda job: job['cache_key'])
    else:
        view_camera = dict(camera)
        requested = set(camera.get('job_keys') or [])
        jobs = [
            job for job in base.discover_scene_texture_jobs(scene_path)
            if not requested or job['cache_key'] in requested]
    retained = []
    try:
        try:
            from tqdm import tqdm
            iterator = tqdm(
                jobs, desc=('Building whole-slope textured result'
                            if whole_scene_context else
                            'Building selected textured result'),
                unit='tile')
        except ImportError:
            iterator = jobs
        for job in iterator:
            for textured_mesh, texture in _textured_job_meshes(
                    job, view_camera, viewer_resolution,
                    texture_dimension):
                retained.append((textured_mesh, texture))
                plotter.add_mesh(
                    textured_mesh, texture=texture, lighting=False,
                    show_edges=False)
        if not retained:
            mesh = _sample_polydata(sample)
            plotter.add_mesh(
                mesh, scalars='texture_rgb', rgb=True, preference='cell',
                lighting=False)
        capture_mpp = float(camera.get(
            'metres_per_pixel', 2 * float(camera['parallel_scale'])
            / max(capture_resolution[1], 1)))
        viewer_mpp = (2 * float(view_camera['parallel_scale'])
                      / max(viewer_resolution[1], 1))
        trace_radius = max(
            capture_mpp * skeleton_display_width_px / 2.0,
            viewer_mpp * 0.8, 0.002)
        for trace in geometry:
            points = np.asarray(trace['points'], dtype=np.float32)
            if len(points) >= 2:
                line = pv.lines_from_points(points, close=False)
                # A thin tube prevents z-fighting with the textured surface
                # while preserving the depth-backprojected centreline.
                plotter.add_mesh(
                    line.tube(radius=trace_radius), color='red',
                    lighting=False)
            elif len(points) == 1:
                plotter.add_points(
                    points, color='red',
                    point_size=max(5, skeleton_display_width_px),
                    render_points_as_spheres=True)
        network_count = len({
            int(trace.get('component_id', 0)) for trace in geometry})
        plotter.add_text(
            f'Depth-backprojected fracture networks: {network_count} '
            f'({len(geometry)} trace segments)',
            position='upper_left', font_size=10, color='white')
        _apply_camera(plotter, view_camera)
        plotter.renderer.ResetCameraClippingRange()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save before entering the native interactor. Windows destroys the
        # render window when its X button is used, making a later screenshot
        # unreliable.
        if interactive:
            # On-screen PyVista plotters require one public ``show`` call
            # before ``screenshot``; interactive=False renders and returns
            # immediately while keeping the native window alive.
            plotter.show(auto_close=False, interactive=False)
        plotter.screenshot(str(output_path))
        if not output_path.is_file():
            raise OSError(f'Could not save textured 3D result: {output_path}')
        if interactive:
            print(
                'Opening whole-slope textured 3D result. '
                'Close the window to finish...')
            plotter.show(auto_close=True, interactive=True)
    finally:
        try:
            plotter.close()
        finally:
            _OBJ_MESH_CACHE.clear()
    return output_path


def run_virtual_camera_fracture_pipeline(
        scene_path: PathLike,
        model_config: PathLike,
        checkpoint: PathLike,
        output_dir: PathLike = 'fracture_3d_pipeline',
        device: str = 'auto',
        capture_resolution: Tuple[int, int] = (3072, 1536),
        reuse_virtual_camera: bool = True,
        lock_camera_normal: bool = True,
        max_projected_metres_per_pixel: Optional[float] = None,
        min_surface_coverage: float = 0.10,
        overwrite_capture: bool = False,
        overwrite_inference: bool = False,
        max_texture_dimension: int = 4096,
        window_size: Tuple[int, int] = (256, 256),
        overlap_ratio: float = 0.4,
        inference_batch_size: int = 4,
        num_classes: int = 2,
        min_component_pixels: int = 20,
        min_trace_length_m: float = 0.10,
        skeleton_display_width_px: int = 9,
        use_amp: bool = True,
        show_interactive_3d: bool = True) -> Optional[Dict]:
    """One-click virtual-camera capture, voting, and depth back-projection."""
    import pandas as pd
    import torch
    from mmseg.apis import inference_model, init_model

    capture_resolution = tuple(map(int, capture_resolution))
    window_size = tuple(map(int, window_size))
    if len(capture_resolution) != 2 or min(capture_resolution) < 16:
        raise ValueError('capture_resolution must contain two values >= 16.')
    if len(window_size) != 2 or min(window_size) < 1:
        raise ValueError('window_size must contain two positive values.')
    if not 0 <= overlap_ratio < 1:
        raise ValueError('overlap_ratio must be in [0, 1).')
    if max_texture_dimension < 16:
        raise ValueError('max_texture_dimension must be at least 16.')
    if inference_batch_size < 1:
        raise ValueError('inference_batch_size must be positive.')
    if num_classes < 2:
        raise ValueError('num_classes must be at least 2.')
    if min_component_pixels < 1 or min_trace_length_m < 0:
        raise ValueError(
            'Component pixels must be positive and trace length non-negative.')
    skeleton_display_width_px = _validate_skeleton_display_width(
        skeleton_display_width_px)
    if not 0 <= min_surface_coverage <= 1:
        raise ValueError('min_surface_coverage must be in [0, 1].')
    if (max_projected_metres_per_pixel is not None
            and max_projected_metres_per_pixel <= 0):
        raise ValueError(
            'max_projected_metres_per_pixel must be positive or None.')
    scene_path = Path(scene_path).expanduser().resolve()
    model_config = Path(model_config).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    for required_path, label in (
            (scene_path, 'scene'), (model_config, 'model config'),
            (checkpoint, 'checkpoint')):
        if not required_path.is_file():
            raise FileNotFoundError(
                f'{label} file does not exist: {required_path}')

    output_dir = Path(output_dir).expanduser().resolve()
    camera = (load_last_virtual_camera(output_dir)
              if reuse_virtual_camera else None)
    if camera is not None:
        saved_resolution = tuple(map(
            int, camera.get('resolution', capture_resolution)))
        saved_aspect = saved_resolution[0] / max(saved_resolution[1], 1)
        requested_aspect = (
            capture_resolution[0] / max(capture_resolution[1], 1))
        reused_mpp = (
            2 * float(camera.get('parallel_scale', float('nan')))
            / capture_resolution[1])
        if camera.get('camera_schema_version') != CAMERA_SCHEMA_VERSION:
            camera = None
        elif camera.get('scene_path') != str(scene_path):
            camera = None
        elif lock_camera_normal and not camera.get('normal_locked', False):
            camera = None
        elif not np.isclose(saved_aspect, requested_aspect, rtol=0, atol=1e-6):
            print(
                'Saved camera aspect ratio differs from capture_resolution; '
                'opening the selector again.')
            camera = None
        elif (max_projected_metres_per_pixel is not None
              and (not np.isfinite(reused_mpp)
                   or reused_mpp > max_projected_metres_per_pixel)):
            print(
                'Saved camera is too coarse for the configured metre/pixel '
                'limit; opening the selector again.')
            camera = None
    if camera is None:
        camera = select_virtual_camera(
            scene_path, output_dir, capture_resolution=capture_resolution,
            detection_window_size=window_size,
            preview_texture_dimension=min(max_texture_dimension, 1024),
            lock_camera_normal=lock_camera_normal,
            max_projected_metres_per_pixel=(
                max_projected_metres_per_pixel))
    if camera is None:
        return None
    camera['resolution'] = list(capture_resolution)
    camera['metres_per_pixel'] = (
        2 * float(camera['parallel_scale']) / capture_resolution[1])
    camera['window_projected_size_m'] = [
        float(window_size[0] * camera['metres_per_pixel']),
        float(window_size[1] * camera['metres_per_pixel'])]
    capture = capture_virtual_camera_view(
        scene_path, camera, output_dir,
        max_texture_dimension=max_texture_dimension,
        overwrite=overwrite_capture)
    camera['job_keys'] = list(capture.get('rendered_job_keys', []))
    base._atomic_save_json(
        output_dir / 'virtual_camera' / 'last_camera.json', camera)
    # Keep parsed OBJ arrays until the whole-slope result is rendered. This
    # avoids reparsing the 3MX production a second time after inference.
    rgb_bgr = cv2.imread(capture['rgb'], cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise OSError(f'Could not read virtual-camera image: {capture["rgb"]}')
    with np.load(capture['xyz']) as xyz_archive:
        world_xyz = xyz_archive['xyz']
    if world_xyz.shape[:2] != rgb_bgr.shape[:2]:
        raise ValueError(
            'RGB and world-XYZ dimensions differ: '
            f'{rgb_bgr.shape[:2]} versus {world_xyz.shape[:2]}')
    surface_mask = np.isfinite(world_xyz).all(axis=2)
    surface_coverage = float(surface_mask.mean())
    if surface_coverage < min_surface_coverage:
        raise ValueError(
            'The selected camera contains too much empty background: '
            f'{surface_coverage:.1%} surface coverage, required at least '
            f'{min_surface_coverage:.1%}. Reselect a tighter rectangle.')
    repository_root = Path(__file__).resolve().parents[1]
    inference_spec = dict(
        inference_algorithm_version=INFERENCE_ALGORITHM_VERSION,
        capture_fingerprint=capture['fingerprint'],
        model_config=_resolved_config_signature(model_config),
        checkpoint=_file_signature(checkpoint),
        model_implementation=_directory_signature(
            repository_root / 'mmseg' / 'models'),
        voting_implementation=_file_signature(
            repository_root / 'tools' / 'fracture_3d_analysis.py'),
        window_size=list(window_size),
        overlap_ratio=float(overlap_ratio),
        num_classes=int(num_classes),
        inference_batch_size=int(inference_batch_size),
        fast_tensor_inference=True,
        force_whole_inference=True,
        use_amp=bool(use_amp))
    inference_fingerprint = _stable_fingerprint(inference_spec)
    prediction_path = (Path(capture['directory'])
                       / f'camera_prediction_{inference_fingerprint}.npy')
    voting_report_path = (Path(capture['directory'])
                          / f'voting_report_{inference_fingerprint}.json')
    reuse_prediction = (
        prediction_path.is_file() and voting_report_path.is_file()
        and not overwrite_capture and not overwrite_inference)
    model = None
    if reuse_prediction:
        try:
            prediction = np.load(prediction_path)
            voting = json.loads(
                voting_report_path.read_text(encoding='utf-8'))
            if (prediction.shape != rgb_bgr.shape[:2]
                    or not np.issubdtype(prediction.dtype, np.integer)
                    or prediction.min(initial=0) < 0
                    or prediction.max(initial=0) >= num_classes
                    or voting.get('method') != 'majority_vote'):
                reuse_prediction = False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            reuse_prediction = False
    if not reuse_prediction:
        print(
            f'Running majority-vote inference '
            f'({int(surface_mask.sum())} visible surface pixels)...')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config = base._prepare_inference_config(
            model_config, None, True, force_whole_inference=True)
        model = init_model(config, str(checkpoint), device=device)
        prediction, voting = base.infer_regions_by_majority_voting(
            model=model,
            image=rgb_bgr,
            inference_model=inference_model,
            selection_mask=surface_mask,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_classes=num_classes,
            batch_size=inference_batch_size,
            show_progress=True,
            fast_tensor_inference=True,
            use_amp=use_amp)
        base._atomic_save_npy(prediction_path, prediction)
        base._atomic_save_json(voting_report_path, voting)
    # VTK needs GPU memory for the complete textured slope. Release the MMSeg
    # model before constructing that view; predictions are already on CPU.
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records, geometry, accepted_skeleton = _measure_camera_prediction(
        prediction, world_xyz, min_component_pixels, min_trace_length_m)
    record_columns = [
        'tile_name', 'texture_name', 'component_id', 'segment_id',
        'skeleton_pixels', 'length_3d_m', 'centroid_e', 'centroid_n',
        'centroid_z', 'strike_deg', 'trace_inclination_deg']
    fractures = pd.DataFrame(records, columns=record_columns)
    measurement_spec = dict(
        measurement_algorithm_version=MEASUREMENT_ALGORITHM_VERSION,
        inference_fingerprint=inference_fingerprint,
        min_component_pixels=int(min_component_pixels),
        min_trace_length_m=float(min_trace_length_m),
        skeleton_connectivity=8,
        depth_jump_factor=8.0,
        junctions_split_into_segments=True)
    measurement_fingerprint = _stable_fingerprint(measurement_spec)
    result_stem = f'fracture_{measurement_fingerprint}'
    metres_per_pixel = float(capture['metres_per_pixel'])
    result_texture_dimension = min(max_texture_dimension, 1024)
    visualization_spec = dict(
        visualization_algorithm_version=VISUALIZATION_ALGORITHM_VERSION,
        measurement_fingerprint=measurement_fingerprint,
        skeleton_display_width_px=skeleton_display_width_px,
        skeleton_measurement_width_px=1,
        display_dilation_kernel='ellipse',
        three_dimensional_radius_m=float(
            metres_per_pixel * skeleton_display_width_px / 2.0),
        texture_dimension=int(result_texture_dimension),
        whole_scene_context=True,
        display_only=True)
    visualization_fingerprint = _stable_fingerprint(visualization_spec)
    visualization_stem = (
        f'{result_stem}_display_{visualization_fingerprint}')
    csv_path = Path(capture['directory']) / f'{result_stem}_parameters_3d.csv'
    fractures.to_csv(csv_path, index=False)
    traces_path = Path(capture['directory']) / f'{result_stem}_traces_3d.json'
    base._atomic_save_json(traces_path, geometry)
    filtered_prediction_path = (
        Path(capture['directory'])
        / f'{result_stem}_accepted_skeleton.npy')
    base._atomic_save_npy(
        filtered_prediction_path, accepted_skeleton.astype(np.uint8))
    accepted_skeleton_mask_path = (
        Path(capture['directory'])
        / f'{result_stem}_accepted_skeleton_1px.png')
    skeleton_display_mask_path = (
        Path(capture['directory'])
        / f'{visualization_stem}_skeleton_mask.png')
    if not cv2.imwrite(
            str(accepted_skeleton_mask_path),
            accepted_skeleton.astype(np.uint8) * 255):
        raise OSError(
            'Could not save one-pixel skeleton mask: '
            f'{accepted_skeleton_mask_path}')
    thick_skeleton = _thicken_skeleton(
        accepted_skeleton, skeleton_display_width_px)
    if not cv2.imwrite(
            str(skeleton_display_mask_path),
            thick_skeleton.astype(np.uint8) * 255):
        raise OSError(
            f'Could not save thick skeleton mask: '
            f'{skeleton_display_mask_path}')
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    raw_prediction_mask_path = (
        Path(capture['directory']) / f'{result_stem}_raw_prediction.png')
    surface_mask_path = (
        Path(capture['directory']) / f'{result_stem}_surface_mask.png')
    if not cv2.imwrite(
            str(raw_prediction_mask_path),
            ((prediction > 0) & surface_mask).astype(np.uint8) * 255):
        raise OSError(
            f'Could not save raw prediction mask: {raw_prediction_mask_path}')
    if not cv2.imwrite(
            str(surface_mask_path), surface_mask.astype(np.uint8) * 255):
        raise OSError(f'Could not save surface mask: {surface_mask_path}')
    raw_overlay_path = _save_raw_prediction_overlay(
        rgb, prediction, surface_mask,
        Path(capture['directory']) / f'{result_stem}_raw_overlay.png')
    overlay_path = _save_camera_overlay(
        rgb, accepted_skeleton,
        Path(capture['directory'])
        / f'{visualization_stem}_skeleton_overlay.png',
        display_width_px=skeleton_display_width_px)
    result_3d_path = (Path(capture['directory'])
                      / f'{visualization_stem}_textured_3d.png')
    stride = voting.get('stride') or [
        max(1, int(window_size[0] * (1 - overlap_ratio))),
        max(1, int(window_size[1] * (1 - overlap_ratio)))]
    fracture_count = len({
        int(record['component_id']) for record in records})
    report = dict(
        method='virtual_camera_depth_backprojection',
        inference_fingerprint=inference_fingerprint,
        inference_spec=inference_spec,
        measurement_fingerprint=measurement_fingerprint,
        measurement_spec=measurement_spec,
        visualization_fingerprint=visualization_fingerprint,
        visualization_spec=visualization_spec,
        surface_coverage=surface_coverage,
        fracture_count=fracture_count,
        trace_segment_count=len(records),
        capture=capture,
        voting=voting,
        camera_scale=dict(
            projected_metres_per_pixel=metres_per_pixel,
            window_projected_size_m=[
                float(window_size[0] * metres_per_pixel),
                float(window_size[1] * metres_per_pixel)],
            stride_projected_size_m=[
                float(stride[0] * metres_per_pixel),
                float(stride[1] * metres_per_pixel)],
            note=('Orthographic camera-plane scale; distances measured from '
                  'depth-backprojected XYZ remain true 3D distances.')),
        filters=dict(
            min_component_pixels=int(min_component_pixels),
            min_trace_length_m=float(min_trace_length_m)),
        outputs=dict(
            camera_rgb=capture['rgb'], camera_depth=capture['depth'],
            camera_world_xyz=capture['xyz'], prediction=str(prediction_path),
            raw_prediction_mask=str(raw_prediction_mask_path),
            raw_prediction_overlay=str(raw_overlay_path),
            surface_mask=str(surface_mask_path),
            accepted_skeleton=str(filtered_prediction_path),
            accepted_skeleton_mask=str(accepted_skeleton_mask_path),
            skeleton_display_mask=str(skeleton_display_mask_path),
            skeleton_overlay=str(overlay_path),
            overlay=str(overlay_path), csv=str(csv_path),
            traces_3d=str(traces_path),
            textured_3d=str(result_3d_path)))
    statistical_paths = save_fracture_statistical_plots(
        fractures, Path(capture['directory']), stem=result_stem,
        fracture_count=fracture_count)
    report['outputs'].update(statistical_paths)
    report_path = (Path(capture['directory'])
                   / f'{visualization_stem}_report.json')
    base._atomic_save_json(report_path, report)
    result = dict(
        fractures=fractures, report=report, trace_geometry=geometry,
        paths={**report['outputs'], 'report': str(report_path)},
        camera=camera)
    print(
        f'Fracture analysis saved: {report_path} '
        f'({fracture_count} networks, {len(records)} trace segments)')
    try:
        if (show_interactive_3d or overwrite_capture or overwrite_inference
                or not result_3d_path.is_file()):
            _show_virtual_camera_result(
                scene_path, output_dir, geometry, camera, result_3d_path,
                texture_dimension=result_texture_dimension,
                interactive=show_interactive_3d,
                whole_scene_context=True,
                skeleton_display_width_px=skeleton_display_width_px)
        else:
            _OBJ_MESH_CACHE.clear()
    except Exception:
        _OBJ_MESH_CACHE.clear()
        raise
    return result
