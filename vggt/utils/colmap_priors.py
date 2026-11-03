"""Read camera priors in the exact selected RGB tensor order."""
from pathlib import Path

import numpy as np
import torch

from .colmap_readers import (
    read_cameras_binary, read_cameras_text, read_images_binary, read_images_text,
)


def load_colmap_priors(scene_dir, sparse_subdir, paths, transforms, original_sizes):
    scene_dir = Path(scene_dir)
    sparse_dir = scene_dir / 'sparse' / sparse_subdir
    if all((sparse_dir / name).is_file() for name in ('cameras.bin', 'images.bin')):
        cameras = read_cameras_binary(sparse_dir / 'cameras.bin')
        registered = read_images_binary(sparse_dir / 'images.bin')
        model_format = 'binary'
    elif all((sparse_dir / name).is_file() for name in ('cameras.txt', 'images.txt')):
        cameras = read_cameras_text(sparse_dir / 'cameras.txt')
        registered = read_images_text(sparse_dir / 'images.txt')
        model_format = 'text'
    else:
        raise FileNotFoundError(
            f'TTO requires cameras.txt + images.txt, or cameras.bin + images.bin in {sparse_dir}'
        )
    by_name = {}
    for image in registered.values():
        name = image.name.replace('\\', '/')
        if name in by_name:
            raise ValueError(f'Duplicate COLMAP image name: {name}')
        by_name[name] = image
    names = [p.relative_to(scene_dir / 'images').as_posix() for p in paths]
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f'Selected input images missing from {sparse_dir}: {missing}')
    extrinsics, intrinsics = [], []
    for name, transform, (ih, iw) in zip(names, transforms, original_sizes):
        image = by_name[name]
        camera = cameras[image.camera_id]
        if camera.model == 'PINHOLE':
            fx, fy, cx, cy = camera.params
        elif camera.model == 'SIMPLE_PINHOLE':
            fx, cx, cy = camera.params
            fy = fx
        else:
            raise ValueError(
                f'Camera {camera.id} uses {camera.model}; provide undistorted PINHOLE or SIMPLE_PINHOLE images/cameras'
            )
        if min(camera.width, camera.height, fx, fy) <= 0:
            raise ValueError(f'Invalid camera dimensions or focal lengths for {name}')
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        # Account for resized image files as well as center cropping/padding.
        file_resize = np.diag([iw / camera.width, ih / camera.height, 1.0])
        intrinsics.append(transform @ file_resize @ intrinsic)
        qvec = np.asarray(image.qvec, dtype=np.float64)
        norm = np.linalg.norm(qvec)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f'Invalid COLMAP quaternion for {name}')
        image = image._replace(qvec=qvec / norm)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = image.qvec2rotmat()
        extrinsic[:3, 3] = image.tvec
        extrinsics.append(extrinsic)
    raw = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)
    if not np.isfinite(raw).all() or not np.isfinite(intrinsics).all():
        raise ValueError('COLMAP cameras contain non-finite values')
    # Same first-camera/median-distance normalization used by demo_colmap.py.
    normalized = raw @ np.linalg.inv(raw[0])
    centers = np.linalg.inv(normalized)[:, :3, 3]
    # torch.median uses the lower middle entry for an even number of views.
    distances = np.sort(np.linalg.norm(centers, axis=-1))
    scale = max(float(distances[(len(distances)-1)//2]), 0.1)
    normalized[:, :3, 3] /= scale
    return {
        'extrinsics': torch.from_numpy(normalized.astype(np.float32))[None],
        'intrinsics': torch.from_numpy(intrinsics.astype(np.float32))[None],
        'original_extrinsics': raw.astype(np.float32),
        'image_names': names,
        'sparse_directory': str(sparse_dir),
        'format': model_format,
        'normalization_scale': scale,
    }
