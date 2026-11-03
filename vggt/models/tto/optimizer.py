"""Optional per-scene LoRA optimization using COLMAP camera priors.

Adapted from the source project's vggt.models.tco implementation. Gaussian
construction runs once after optimization; the pretrained checkpoint is frozen.
"""
from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from addict import Dict

from .backbone import TCOCheckpointBackboneMixin
from .constraints import camera_pose_energy, intrinsics_energy
from .lora import TransformerLoRA


@dataclass
class TTOConfig:
    enabled: bool = False
    steps: int = 20
    lr: float = 5e-4
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.0
    lambda_pose: float = 1.0
    pose_translation_weight: float = 1.0
    lambda_intrinsics: float = 0.01
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    target_modules: Optional[list[str]] = None
    save_lora: bool = True

    def validate(self):
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError('tto.steps must be a positive integer when TTO is enabled')
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError('tto.rank must be a positive integer')
        for name in ('lr', 'alpha', 'grad_clip'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'tto.{name} must be finite and positive')
        for name in ('lambda_pose', 'lambda_intrinsics', 'pose_translation_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'tto.{name} must be finite and nonnegative')
        if self.lambda_pose == self.lambda_intrinsics == 0:
            raise ValueError('TTO requires at least one positive camera loss weight')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('tto.dropout must be in [0, 1)')
        if self.target_modules is not None and not self.target_modules:
            raise ValueError('tto.target_modules must be null or a nonempty list')


class CameraOptimization(TCOCheckpointBackboneMixin):
    """Use the cached camera-only path without the inference-mode wrapper."""
    def __init__(self, model, config):
        self.model = model.model
        self._maybe_normalize_images = model._maybe_normalize_images
        self.tco_config = config

    def predict(self, cache):
        feats = self.forward_tco_cached_backbone(cache, require_dense_features=False)
        output = self.model._process_camera_estimation(feats, cache.height, cache.width, Dict())
        return output.extrinsics, output.intrinsics


def run_tto(model, images, priors, config, *, valid_image_mask=None,
            ref_view_strategy='saddle_balanced'):
    """Adapt LoRA, produce final Gaussian predictions, then remove adapters.

    Call outside torch.inference_mode(): optimization needs autograd. Original
    parameter values and requires_grad flags are restored, including on errors.
    """
    config.validate()
    if torch.is_inference_mode_enabled():
        raise RuntimeError('TTO must be called outside torch.inference_mode()')
    if model.training:
        raise ValueError('TTO expects the pretrained model in eval mode')
    prior_extrinsics = priors['extrinsics'].to(images.device)
    prior_intrinsics = priors['intrinsics'].to(images.device)
    if images.shape[-4] < 2 and config.lambda_pose > 0 and config.lambda_intrinsics == 0:
        raise ValueError('Pose-only TTO needs at least two views')
    transformer = model.model.backbone.pretrained
    start = int(transformer.alt_start)
    suffix = torch.nn.ModuleDict({str(i): transformer.blocks[i] for i in range(start, len(transformer.blocks))})
    lora = TransformerLoRA(config.target_modules) if config.target_modules is not None else TransformerLoRA()
    previous_flags = [(p, p.requires_grad) for p in model.parameters()]
    optimizer = cache = None
    history = []
    try:
        names = lora.inject(suffix, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        lora.set_requires_grad(True)
        parameters = list(lora.parameters())
        optimizer = torch.optim.Adam(parameters, lr=config.lr)
        runner = CameraOptimization(model, config)
        with torch.enable_grad():
            cache = runner.prepare_tco_backbone_cache(
                images, extrinsics=None, intrinsics=None, ref_view_strategy=ref_view_strategy)
            def evaluate():
                extrinsics, intrinsics = runner.predict(cache)
                rotation, translation = camera_pose_energy(extrinsics, prior_extrinsics)
                intrinsic = intrinsics_energy(intrinsics, prior_intrinsics, images.shape[-2:])
                total = config.lambda_pose * (rotation + config.pose_translation_weight * translation)
                total = total + config.lambda_intrinsics * intrinsic
                return total, {'pose_rotation': rotation, 'pose_translation': translation, 'intrinsics': intrinsic}
            for step in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                loss, components = evaluate()
                if not loss.requires_grad or not torch.isfinite(loss):
                    raise FloatingPointError(f'TTO step {step}: loss is non-finite or has no gradient')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                record = {key: float(value.detach()) for key, value in components.items()}
                record.update(step=step+1, total=float(loss.detach()), grad_norm=float(grad_norm))
                history.append(record)
                print(f'TTO {step+1:03d}/{config.steps:03d}: loss={record["total"]:.6f}', flush=True)
                del loss, components
            lora.set_requires_grad(False)
            for item in lora.injected:
                item.adapter.eval()
            loss, components = evaluate()
            final = {key: float(value.detach()) for key, value in components.items()}
            final['total'] = float(loss.detach())
            del loss, components
        adapter_state = {k: v.cpu() for k, v in lora.state_dict().items()} if config.save_lora else None
        adapted = any(bool(torch.count_nonzero(item.adapter.lora_B).item()) for item in lora.injected)
        # Release optimization graphs/state before the memory-heavy Gaussian pass.
        cache = None
        optimizer.zero_grad(set_to_none=True)
        optimizer = None
        if images.is_cuda:
            torch.cuda.empty_cache()
        with torch.inference_mode():
            predictions = model(images, valid_image_mask=valid_image_mask, ref_view_strategy=ref_view_strategy)
        result = {'config': asdict(config), 'history': history, 'final_loss': final,
                  'adapted_layers': names, 'adapters_updated': adapted,
                  'lora_state_dict': adapter_state}
        return predictions, result
    finally:
        cache = optimizer = None
        for parameter in lora.parameters():
            parameter.grad = None
        lora.remove(merge=False)
        for parameter, enabled in previous_flags:
            parameter.requires_grad_(enabled)
