#!/usr/bin/env python
"""Run OctreeSplat on a COLMAP scene with explicit input/output paths."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from vggt.models.depthanything3 import DepthAnything3
from vggt.utils.gsply_helpers import save_gaussian_ply

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_images(cfg, scene_dir):
    image_dir = project_path(scene_dir) / 'images'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'COLMAP scene images directory does not exist: {image_dir}')
    paths = sorted(p for p in image_dir.glob(cfg.input.glob)
                   if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'})
    if cfg.input.indices is not None:
        if any(i < 0 or i >= len(paths) for i in cfg.input.indices):
            raise ValueError('input.indices is outside the image list')
        paths = [paths[i] for i in cfg.input.indices]
    else:
        if cfg.input.stride < 1:
            raise ValueError('input.stride must be positive')
        paths = paths[::cfg.input.stride]
        if cfg.input.max_images is not None:
            if cfg.input.max_images < 1:
                raise ValueError('input.max_images must be positive or null')
            paths = paths[:cfg.input.max_images]
    if not paths:
        raise ValueError(f'No images found in {image_dir}')
    h, w, patch = int(cfg.preprocess.height), int(cfg.preprocess.width), int(cfg.preprocess.patch_size)
    if min(h, w, patch) <= 0 or h % patch or w % patch or patch != 14:
        raise ValueError('Input dimensions must be positive multiples of patch_size=14')
    tensors, masks, transforms, sizes = [], [], [], []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert('RGBA')
            background = Image.new('RGBA', image.size, (255,255,255,255))
            image = Image.alpha_composite(background, image).convert('RGB')
        iw, ih = image.size
        if cfg.preprocess.mode == 'center_crop':
            scale = max(w / iw, h / ih)
            rw, rh = int(np.ceil(iw * scale)), int(np.ceil(ih * scale))
            left, top = (rw-w)//2, (rh-h)//2
            image = image.resize((rw,rh), Image.Resampling.BICUBIC).crop((left,top,left+w,top+h))
            dx, dy = -left, -top
            mask = torch.ones((h,w), dtype=torch.bool)
        elif cfg.preprocess.mode == 'fit_pad':
            scale = min(w / iw, h / ih)
            rw, rh = max(1, round(iw*scale)), max(1, round(ih*scale))
            dx, dy = (w-rw)//2, (h-rh)//2
            canvas = Image.new('RGB',(w,h))
            canvas.paste(image.resize((rw,rh), Image.Resampling.BICUBIC),(dx,dy))
            image = canvas
            mask = torch.zeros((h,w), dtype=torch.bool)
            mask[dy:dy+rh,dx:dx+rw] = True
        else:
            raise ValueError('preprocess.mode must be center_crop or fit_pad')
        transforms.append([[rw/iw,0,dx],[0,rh/ih,dy],[0,0,1]])
        sizes.append([ih,iw])
        tensors.append(torch.from_numpy(np.asarray(image,dtype=np.float32)/255).permute(2,0,1))
        masks.append(mask)
    return torch.stack(tensors), torch.stack(masks)[None], paths, np.asarray(transforms), np.asarray(sizes)


def load_model(cfg, device_name='cuda', bf16=False):
    checkpoint = project_path(cfg.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('OctreeSplat sparse convolutions require an available CUDA device')
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = DepthAnything3(OmegaConf.to_container(cfg.model,resolve=True), cfg.manual_metric_divisor)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    state = state.get('model',state)
    # DDP prefixes are optional; only removed backend tensors may be ignored.
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    expected = model.state_dict()
    removed = [k for k in state if k.startswith(('model.backend.', 'model.cam_enc.'))
               or (k.startswith('model.head.scratch.') and '_aux' in k)]
    unexpected = set(state)-set(expected)-set(removed)
    missing = set(expected)-set(state)
    mismatched = [k for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError(f'Checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shape={mismatched}')
    model.load_state_dict({k:v for k,v in state.items() if k in expected},strict=True)
    del state, expected
    model.eval().to(device).to(dtype)
    print(f'Loaded checkpoint; ignored {len(removed)} unused backend/camera-conditioning/ray-head tensors', flush=True)
    return model, device


def export(predictions, cfg, paths, transforms, original_sizes, output_dir):
    out = project_path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,out/'inference.yaml',resolve=True)
    if cfg.output.save_npz:
        arrays = {k:predictions[k][0].detach().float().cpu().numpy()
                  for k in ('depth','depth_conf','extrinsics','intrinsics')}
        arrays.update(image_names=np.array([p.name for p in paths]), image_transform=transforms,
                      original_sizes=original_sizes)
        for key in ('octree_leaf_coords','octree_leaf_levels'):
            if key in predictions:
                arrays[key] = predictions[key][0].detach().cpu().numpy()
        np.savez_compressed(out/'predictions.npz',**arrays)
    if cfg.output.save_ply:
        confidence = predictions.get('gaussian_voxel_depth_conf')
        mask = None if confidence is None else confidence[0] > cfg.output.confidence_threshold
        options = dict(ctx_depth=predictions['depth'][0], save_sh_dc_only=cfg.output.save_sh_dc_only,
                       prune_by_depth_percent=1.0, prune_border_gs=False, gaussian_conf_mask=mask,
                       conf_mask=(predictions['depth_conf'][0] > cfg.output.confidence_threshold), inv_opacity=True)
        save_gaussian_ply(predictions['gs_world'],str(out/'gaussians.ply'),**options)
        if cfg.output.save_pyramid:
            if 'gs_pyramid_world' not in predictions:
                raise ValueError('output.save_pyramid requires model.backend_gs_decoder.inference_mode=all')
            options['gaussian_conf_mask'] = None
            for level, gs in enumerate(predictions['gs_pyramid_world']):
                save_gaussian_ply(gs,str(out/f'gaussians_level{level}.ply'),**options)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/inference.yaml'))
    parser.add_argument('--input', required=True, help='COLMAP scene folder: images/ and sparse/gt/ (for TTO)')
    parser.add_argument('--output', required=True, help='Folder for inference results')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 instead of the default FP32')
    parser.add_argument('--device', default='cuda', help='CUDA device, e.g. cuda:0')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--allow-tf32', action='store_true', help='Allow TF32 for FP32 operations')
    parser.add_argument('--ref-view-strategy', default='saddle_balanced', help='Backbone reference-view strategy')
    tto_switch = parser.add_mutually_exclusive_group()
    tto_switch.add_argument('--tto', dest='tto', action='store_true', help='Enable COLMAP-prior LoRA TTO')
    tto_switch.add_argument('--no-tto', dest='tto', action='store_false', help='Disable TTO')
    parser.set_defaults(tto=None)
    parser.add_argument('overrides', nargs='*', help='Optional YAML dotlist overrides, e.g. input.max_images=3')
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(Path(args.config).expanduser()),OmegaConf.from_dotlist(args.overrides))
    if args.tto is not None:
        cfg = OmegaConf.merge(cfg, {'tto': {'enabled': args.tto}})
    OmegaConf.resolve(cfg)
    tto_enabled = bool(cfg.get('tto', {}).get('enabled', False))
    if cfg.output.save_pyramid and cfg.model.backend_gs_decoder.inference_mode != 'all':
        raise ValueError('save_pyramid requires inference_mode=all')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    images, mask, paths, transforms, sizes = load_images(cfg, args.input)
    print(f'Input: {len(paths)} images, {tuple(images.shape[-2:])}',flush=True)
    priors = tto_config = None
    if tto_enabled:
        from vggt.models.tto.optimizer import TTOConfig
        from vggt.utils.colmap_priors import load_colmap_priors
        tto_config = TTOConfig(**OmegaConf.to_container(cfg.tto, resolve=True))
        tto_config.validate()
        priors = load_colmap_priors(project_path(args.input), cfg.input.get('sparse_subdir', 'gt'),
                                   paths, transforms, sizes)
        print(f'TTO camera priors: {priors["sparse_directory"]} ({priors["format"]})', flush=True)
    model, device = load_model(cfg, args.device, args.bf16)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    images = images.to(device=device, dtype=next(model.model.backbone.parameters()).dtype)
    mask = mask.to(device)
    tto_result = None
    if tto_enabled:
        from vggt.models.tto.optimizer import run_tto
        predictions, tto_result = run_tto(model, images, priors, tto_config,
                                           valid_image_mask=mask, ref_view_strategy=args.ref_view_strategy)
    else:
        with torch.inference_mode():
            predictions = model(images, valid_image_mask=mask, ref_view_strategy=args.ref_view_strategy)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    stats = dict(input=str(project_path(args.input)), output=str(project_path(args.output)),
                 device=str(device), dtype='bfloat16' if args.bf16 else 'float32',
                 seed=args.seed, allow_tf32=args.allow_tf32, ref_view_strategy=args.ref_view_strategy,
                 images=len(paths),gaussians=predictions['gs_world'].means.shape[1],seconds=elapsed,
                 peak_gpu_gib=torch.cuda.max_memory_allocated(device)/(1024**3),
                 tto_enabled=tto_enabled, tto_steps=len(tto_result['history']) if tto_result else 0)
    out = export(predictions,cfg,paths,transforms,sizes,args.output)
    if tto_result is not None:
        adapter_state = tto_result.pop('lora_state_dict')
        tto_result['prior'] = {key: priors[key] for key in
                               ('sparse_directory', 'format', 'normalization_scale', 'image_names')}
        if adapter_state is not None:
            torch.save({'state_dict': adapter_state, 'config': tto_result['config'],
                        'checkpoint': str(project_path(cfg.checkpoint)),
                        'image_names': priors['image_names']}, out/'tto_lora.pt')
        (out/'tto.json').write_text(json.dumps(tto_result, indent=2)+'\n')
        np.savez_compressed(out/'tto_camera_priors.npz', image_names=np.asarray(priors['image_names']),
                            extrinsics=priors['extrinsics'][0].numpy(),
                            intrinsics=priors['intrinsics'][0].numpy(),
                            original_extrinsics=priors['original_extrinsics'])
    (out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
