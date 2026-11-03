"""OctreeSplat inference wrapper preserving checkpoint parameter names."""
from typing import Any
import torch
from torch import nn
from omegaconf import OmegaConf
from vggt.models.depth_anything_3.cfg import create_object
from vggt.utils.specs import Gaussians
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

class DepthAnything3(nn.Module):

    def __init__(self, config, manual_metric_divisor=300.0):
        super().__init__()
        self.use_manual_metric_scaling = True
        self.manual_metric_divisor = float(manual_metric_divisor)
        self.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.model = create_object(OmegaConf.create(config))
        if self.model.backend_gs_decoder.manual_metric_divisor != self.manual_metric_divisor:
            raise ValueError('Decoder and wrapper metric divisors must match')

    @torch.inference_mode()
    def forward(self, images, valid_image_mask=None, ref_view_strategy='saddle_balanced'):
        if self.training:
            raise RuntimeError('Call eval() before inference')
        if images.ndim == 4:
            images = images.unsqueeze(0)
        rgb = images
        output = self.model(self._maybe_normalize_images(images), rgb_images=rgb, valid_image_mask=valid_image_mask, ref_view_strategy=ref_view_strategy)
        scale = self._compute_manual_metric_scales(output.depth, output.intrinsics)[1]
        self.model.backend_gs_decoder.prepare_inference(output, rgb, valid_image_mask, scene_scale=scale)
        return self._build_predictions_from_output(output, images.shape[-2:])

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        sparse_conv_fp16 = next(self.model.parameters()).dtype in (torch.bfloat16, torch.float16)
        self._imagenet_mean = self._imagenet_mean.to(*args, **kwargs)
        self._imagenet_std = self._imagenet_std.to(*args, **kwargs)
        dpt_head = getattr(self.model, 'head', None)
        scratch = getattr(dpt_head, 'scratch', None)
        if scratch is not None:
            output_conv2 = getattr(scratch, 'output_conv2', None)
            if output_conv2 is not None:
                output_conv2.float()
        camera_dec = getattr(self.model, 'cam_dec', None)
        if camera_dec is not None:
            camera_dec.float()
        gaussian_decoder = getattr(self.model, 'backend_gs_decoder', None)
        if gaussian_decoder is not None:
            gaussian_decoder.float()
            for module in gaussian_decoder.modules():
                if hasattr(module, 'sparse_conv_fp16'):
                    from vggt.heads.spunet_encoder import set_sparse_conv_fp16
                    set_sparse_conv_fp16(module, sparse_conv_fp16)
        return self

    def _maybe_normalize_images(self, images: torch.Tensor, normalize_images: bool=True) -> torch.Tensor:
        if not normalize_images:
            return images
        img_min = images.amin().item()
        img_max = images.amax().item()
        if img_min >= -0.001 and img_max <= 1.0 + 0.001:
            images = (images - self._imagenet_mean) / self._imagenet_std
        return images

    def _compute_manual_metric_scales(self, depth: torch.Tensor, intrinsics: torch.Tensor):
        if self.manual_metric_divisor <= 0:
            raise ValueError(f'manual_metric_divisor must be > 0, got {self.manual_metric_divisor}')
        focal = 0.5 * (intrinsics[..., 0, 0] + intrinsics[..., 1, 1])
        if depth.ndim == 5:
            depth_scale = (focal / self.manual_metric_divisor)[..., None, None, None]
        elif depth.ndim == 4:
            depth_scale = (focal / self.manual_metric_divisor)[..., None, None]
        else:
            raise ValueError(f'Unexpected depth ndim: {depth.ndim}')
        scene_scale = (focal / self.manual_metric_divisor).mean(dim=1)
        return (depth_scale, scene_scale)

    @staticmethod
    def _scale_extrinsics_translation(
        extrinsics: torch.Tensor,
        scene_scale: torch.Tensor,
    ) -> torch.Tensor:
        scaled_t = extrinsics[..., :3, 3] * scene_scale[:, None, None]
        top = torch.cat([extrinsics[..., :3, :3], scaled_t.unsqueeze(-1)], dim=-1)
        if extrinsics.shape[-2] == 4:
            bottom = extrinsics[..., 3:4, :]
            return torch.cat([top, bottom], dim=-2)
        return top

    @staticmethod
    def _scale_gaussians(gs_world, scene_scale: torch.Tensor):
        if gs_world is None:
            return gs_world
        scale = scene_scale[:, None, None]
        if hasattr(gs_world, 'means') and torch.is_tensor(gs_world.means):
            means = gs_world.means * scale
            scales = gs_world.scales * scale if torch.is_tensor(getattr(gs_world, 'scales', None)) else gs_world.scales
            return Gaussians(means=means, scales=scales, rotations=gs_world.rotations, harmonics=gs_world.harmonics, opacities=gs_world.opacities, features=getattr(gs_world, 'features', None))
        if isinstance(gs_world, dict):
            out = dict(gs_world)
            if torch.is_tensor(gs_world.get('means', None)):
                out['means'] = gs_world['means'] * scale
            if torch.is_tensor(gs_world.get('scales', None)):
                out['scales'] = gs_world['scales'] * scale
            return out
        return gs_world

    def _build_predictions_from_output(self, output, image_hw: tuple[int, int]=None) -> dict[str, Any]:
        depth = output.depth
        if depth.ndim == 4:
            depth = depth.unsqueeze(-1)
        depth_conf = output.get('depth_conf', None)
        gaussian_voxel_depth_conf = output.get('gaussian_voxel_depth_conf', None)
        highres_backend_voxel_point_ratio = output.get('highres_backend_voxel_point_ratio', None)
        if depth_conf is None:
            depth_conf = torch.ones_like(depth[..., 0])
        extrinsics = output.extrinsics
        intrinsics = output.intrinsics
        if self.use_manual_metric_scaling:
            depth_scale, scene_scale = self._compute_manual_metric_scales(depth, intrinsics)
            depth = depth * depth_scale
            extrinsics = self._scale_extrinsics_translation(extrinsics, scene_scale)
        if extrinsics.shape[-2:] == (4, 4):
            extrinsics = extrinsics[..., :3, :]
        pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_hw)
        predictions = {'pose_enc': pose_enc, 'pose_enc_list': [pose_enc], 'extrinsics': extrinsics, 'intrinsics': intrinsics, 'depth': depth, 'depth_conf': depth_conf, 'gaussian_voxel_depth_conf': gaussian_voxel_depth_conf}
        if highres_backend_voxel_point_ratio is not None:
            predictions['highres_backend_voxel_point_ratio'] = highres_backend_voxel_point_ratio
        for key in ('octree_leaf_coords', 'octree_leaf_levels', 'octree_densification_scores'):
            if key in output:
                predictions[key] = output[key]
        if 'gaussians' not in output:
            return predictions
        gs_world = output.gaussians
        if self.use_manual_metric_scaling:
            gs_world = self._scale_gaussians(gs_world, scene_scale)
        gs_world = self._normalize_gaussians_opacity_shape(gs_world)
        predictions['gs_world'] = gs_world
        if 'gaussian_pyramid' in output:
            predictions['gs_pyramid_world'] = [self._scale_gaussians(gs, scene_scale) if self.use_manual_metric_scaling else gs for gs in output['gaussian_pyramid']]
        return predictions
        return predictions

    @staticmethod
    def _normalize_gaussians_opacity_shape(gs_world):
        if gs_world is None:
            return gs_world
        opacities = getattr(gs_world, 'opacities', None)
        if not torch.is_tensor(opacities):
            return gs_world
        if opacities.ndim >= 3 and opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        elif opacities.ndim >= 3:
            opacities = opacities.reshape(opacities.shape[0], opacities.shape[1], -1)[..., 0]
        if hasattr(gs_world, 'opacities'):
            gs_world.opacities = opacities
        elif isinstance(gs_world, dict):
            gs_world['opacities'] = opacities
        return gs_world
