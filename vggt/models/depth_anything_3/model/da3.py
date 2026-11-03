# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from omegaconf import OmegaConf
from vggt.models.depth_anything_3.cfg import create_object
from vggt.models.depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from vggt.models.depth_anything_3.utils.geometry import affine_inverse, unproject_depth
from vggt.models.amb3r.backend import BackEnd

class DepthAnything3Net(nn.Module):
    PATCH_SIZE = 14
    def __init__(self, net, head, cam_dec, backend_dense_head, backend_gs_decoder, backend_option):
        super().__init__()
        for name, config in dict(backbone=net, head=head, cam_dec=cam_dec,
                                 backend_dense_head=backend_dense_head, backend_gs_decoder=backend_gs_decoder).items():
            setattr(self, name, create_object(OmegaConf.create(config)))
        self.backend = BackEnd(voxel_chunk_size=backend_option.get('voxel_chunk_size'))
        self.high_resolution_backend_uptimes = int(backend_option['high_resolution_uptimes'])
        if self.high_resolution_backend_uptimes != self.PATCH_SIZE:
            raise ValueError('This model uses dense-head features: high_resolution_uptimes must equal 14')
        dim = self._get_backend_dense_feature_dim()
        self.backend_dense_feature_proj = (nn.Identity() if dim == self.backend_gs_decoder.out_dim
                                          else nn.Linear(dim, self.backend_gs_decoder.out_dim))

    def forward(self, x, rgb_images, valid_image_mask=None, ref_view_strategy='saddle_balanced'):
        H, W = x.shape[-2:]
        if valid_image_mask is not None:
            if tuple(valid_image_mask.shape) != tuple(x.shape[:2]) + (H, W):
                raise ValueError('valid_image_mask shape must be (B, N, H, W)')
            valid_image_mask = valid_image_mask.to(device=x.device, dtype=torch.bool)
        feats, _ = self.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy=ref_view_strategy)
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            output = self._process_camera_estimation(feats, H, W, output)
        if valid_image_mask is not None:
            output.depth_conf = output.depth_conf.masked_fill(~valid_image_mask, -torch.inf)
        output.depth = output.depth.clamp(min=0.01, max=10.0)
        up = self.high_resolution_backend_uptimes
        points, features, shape = self._prepare_backend_points_and_features(feats, output, None, None, H, W, up)
        target = (shape[0]*up, shape[1]*up)
        colors = self._resize_bt_hwc(rgb_images.permute(0,1,3,4,2), target)
        view_ids = torch.arange(points.shape[1], device=points.device)[None,:,None,None].expand(points.shape[:-1])
        if output.get('depth_conf') is not None:
            keep = self._resize_bt_hw_mask(output.depth_conf >= 0.0, target)
            view_ids = [ids.reshape(-1)[mask.reshape(-1)] for ids, mask in zip(view_ids, keep)]
        points, features, colors, _, confidence = self._filter_backend_inputs_by_confidence(
            points, features, colors, None, output.get('depth_conf'), target, min_conf=0.0)
        sizes = self.backend_gs_decoder.backend_voxel_sizes(output)
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, details = self.backend(points, features, voxel_sizes=sizes, return_voxel_details=True,
                                      colors=colors, depth_conf=confidence, skip_interpolation=True)
        details[0]['octree_points'] = [p.detach().reshape(-1,3) for p in points]
        details[0]['octree_point_view_ids'] = [ids.reshape(-1) for ids in view_ids]
        del points, features, colors, confidence, feats
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self.backend_gs_decoder(output, {'voxel_pyramid': details})
        return output

    def _resize_bt_hwc(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        (batch_size, views, _, _, channels) = tensor.shape
        tensor_resized = F.interpolate(tensor.permute(0, 1, 4, 2, 3).flatten(0, 1), size=target_size, mode='bilinear', align_corners=False)
        tensor_resized = tensor_resized.permute(0, 2, 3, 1).view(batch_size, views, target_size[0], target_size[1], channels)
        return tensor_resized

    def _resize_bt_hwc_min(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        (batch_size, views, height, width, channels) = tensor.shape
        if (height, width) == target_size:
            return tensor
        tensor_2d = tensor.permute(0, 1, 4, 2, 3).flatten(0, 1)
        if target_size[0] <= height and target_size[1] <= width:
            tensor_resized = -F.adaptive_max_pool2d(-tensor_2d, output_size=target_size)
        else:
            tensor_resized = F.interpolate(tensor_2d, size=target_size, mode='bilinear', align_corners=False)
        tensor_resized = tensor_resized.permute(0, 2, 3, 1).view(batch_size, views, target_size[0], target_size[1], channels)
        return tensor_resized

    def _resize_bt_hw_mask(self, tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        (batch_size, views, _, _) = tensor.shape
        invalid_mask = (~tensor.bool()).flatten(0, 1).unsqueeze(1).float()
        invalid_mask_resized = F.adaptive_max_pool2d(invalid_mask, output_size=target_size)
        return invalid_mask_resized.view(batch_size, views, target_size[0], target_size[1]) == 0

    def _filter_backend_inputs_by_confidence(
        self,
        pts_resized: torch.Tensor,
        feat_resized: torch.Tensor,
        colors_resized: torch.Tensor | None,
        image_features_resized: torch.Tensor | None,
        depth_conf: torch.Tensor | None,
        target_size: tuple[int, int],
        min_conf: float=5.0,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor] | None, torch.Tensor | list[torch.Tensor] | None, list[torch.Tensor] | None]:
        if depth_conf is None:
            return (pts_resized, feat_resized, colors_resized, image_features_resized, None)
        conf_keep_mask = self._resize_bt_hw_mask(depth_conf >= min_conf, target_size)
        conf_values = self._resize_bt_hwc_min(depth_conf.unsqueeze(-1), target_size)
        filtered_pts = []
        filtered_feats = []
        filtered_colors = [] if colors_resized is not None else None
        filtered_image_features = [] if image_features_resized is not None else None
        filtered_conf = []
        for batch_idx in range(pts_resized.shape[0]):
            keep_flat = conf_keep_mask[batch_idx].reshape(-1)
            filtered_pts.append(pts_resized[batch_idx].reshape(-1, pts_resized.shape[-1])[keep_flat])
            filtered_feats.append(feat_resized[batch_idx].reshape(-1, feat_resized.shape[-1])[keep_flat])
            filtered_conf.append(conf_values[batch_idx].reshape(-1, conf_values.shape[-1])[keep_flat])
            if filtered_colors is not None:
                filtered_colors.append(colors_resized[batch_idx].reshape(-1, colors_resized.shape[-1])[keep_flat])
            if filtered_image_features is not None:
                filtered_image_features.append(image_features_resized[batch_idx].reshape(-1, image_features_resized.shape[-1])[keep_flat])
        return (filtered_pts, filtered_feats, filtered_colors, filtered_image_features, filtered_conf)

    def _prepare_backend_points_and_features(
        self,
        feats: list[torch.Tensor],
        output: Dict[str, torch.Tensor],
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        H: int,
        W: int,
        upsample_factor: int=2,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[int, int, int, int, int] | None]:
        if self.backend is None:
            return (None, None, None)
        depth_us = output['depth'].unsqueeze(-1)
        ctx_extr = output.get('extrinsics', extrinsics)
        ctx_intr = output.get('intrinsics', intrinsics)
        if ctx_extr is None or ctx_intr is None:
            return (None, None, None)
        with torch.autocast(device_type=depth_us.device.type, enabled=False):
            ctx_c2w = affine_inverse(ctx_extr.float())
            world_points = unproject_depth(depth_us.float(), ctx_intr.float(), c2w=ctx_c2w)
        (ph, pw) = (H // self.PATCH_SIZE, W // self.PATCH_SIZE)
        (batch_size, views, token_count, channels) = feats[-1][0].shape
        backend_feat = feats[-1][0]
        if token_count > ph * pw:
            backend_feat = backend_feat[:, :, -ph * pw:, :]
        backend_feat_reshaped = backend_feat.reshape(batch_size, views, ph, pw, channels)
        target_size = (ph * upsample_factor, pw * upsample_factor)
        with torch.autocast(device_type=world_points.device.type, enabled=False):
            pts_resized = self._resize_bt_hwc(world_points, target_size=target_size)
        use_dense_head_features = upsample_factor == self.PATCH_SIZE and self.backend_dense_feature_proj is not None
        if use_dense_head_features:
            with torch.amp.autocast('cuda', enabled=False):
                dense_feat = self.backend_dense_head(feats, H, W, patch_start_idx=0, chunk_size=8)
                feat_resized = self.backend_dense_feature_proj(dense_feat)
        else:
            raise ValueError('Only dense-head feature inference is supported')
        return (pts_resized, feat_resized, (ph, pw, batch_size, views, token_count))

    def _get_backend_dense_feature_dim(self) -> int | None:
        if self.backend_dense_head is None or not hasattr(self.backend_dense_head, 'scratch'):
            return None
        output_conv1 = getattr(self.backend_dense_head.scratch, 'output_conv1', None)
        return getattr(output_conv1, 'out_channels', None)

    def _process_depth_head(self, feats: list[torch.Tensor], H: int, W: int) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            if 'ray' in output:
                del output.ray
            if 'ray_conf' in output:
                del output.ray_conf
            (c2w, ixt) = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt
        return output
