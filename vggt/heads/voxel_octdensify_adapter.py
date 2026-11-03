"""Three supervised Gaussian maps with optional occupied octree inference."""
from __future__ import annotations
import math
import torch
from torch import nn
import spconv.pytorch as spconv
from vggt.heads.multiscale_spunet_encoder import MultiscaleSpUNetVoxelEncoder, lookup_rows
from vggt.utils.octdensify import gather_gaussians, max_view_counts, pack_scenes, select_leaves, split_fractions
from vggt.heads.voxel_gs_adapter import VoxelGSAdapter
from vggt.models.amb3r.tools.voxel_utils import points_to_voxel_grid

class SparseDensificationScoreHead(nn.Module):
    """One spatial sparse convolution followed by a per-voxel linear score."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.sparse_conv_fp16 = False
        self.conv = spconv.SubMConv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.activation = nn.GELU()
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, features, coords):
        enabled = self.sparse_conv_fp16 and features.is_cuda
        with torch.autocast('cuda', dtype=torch.float16, enabled=enabled):
            scores = self._forward(features, coords, enabled)
        return scores.float()

    def _forward(self, features, coords, use_fp16=False):
        if len(features) != len(coords):
            raise ValueError('One coordinate is required per score-head feature')
        if not len(coords):
            zero = features.sum() * 0.0
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0.0
            return features.new_empty((0,)) + zero
        indices = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1).int()
        spatial_shape = (coords.amax(0).long() + 3).tolist()
        sparse = spconv.SparseConvTensor(features.to(dtype=torch.float16 if use_fp16 else torch.float32), indices, spatial_shape, 1)
        sparse = self.conv(sparse)
        rows = lookup_rows(sparse.indices[:, 1:].long(), coords)
        return self.linear(self.activation(sparse.features[rows])).squeeze(-1)

class VoxelOctDensifyAdapter(nn.Module):

    def __init__(
        self,
        backend_out_dim,
        backbone_dim=128,
        coarse_voxel_size: float=0.008,
        spunet_base_channels=32,
        spunet_channels=(32, 64, 128, 256, 256, 128, 96, 96),
        spunet_layers=(2, 3, 4, 6, 2, 2, 2, 2),
        spunet_sparse_shape_padding=96,
        manual_metric_divisor=None,
        octree_densify_methods=('learned_score', 'learned_score'),
        octree_split_fraction=(0.3, 0.3),
        inference_mode='adaptive',
        octree_split_strategy='per_level',
        densification_score_enabled=False,
        densification_score_hidden_dim=128,
        densification_score_detach_features=True,
        **decoder_options,
    ):
        super().__init__()
        self.out_dim = backend_out_dim
        self.coarse_voxel_size = float(coarse_voxel_size)
        if not math.isfinite(self.coarse_voxel_size) or self.coarse_voxel_size <= 0:
            raise ValueError('coarse_voxel_size must be finite and positive')
        self.multiscale_voxel_sizes = tuple((self.coarse_voxel_size / 2 ** level for level in range(3)))
        self.manual_metric_divisor = manual_metric_divisor
        if manual_metric_divisor is not None and manual_metric_divisor <= 0:
            raise ValueError('manual_metric_divisor must be positive or None')
        self.densification_score_enabled = bool(densification_score_enabled)
        self.densification_score_detach_features = bool(densification_score_detach_features)
        self.octree_densify_methods = self._validate_methods(octree_densify_methods)
        self.octree_split_fraction = split_fractions(octree_split_fraction, len(self.octree_densify_methods))
        if octree_split_strategy not in {'per_level', 'mixed'}:
            raise ValueError('octree_split_strategy must be per_level or mixed')
        self.octree_split_strategy = octree_split_strategy
        if inference_mode not in {'adaptive', 'finest', 'all'}:
            raise ValueError('inference_mode must be adaptive, finest or all')
        self.inference_mode = inference_mode
        self.encoder = MultiscaleSpUNetVoxelEncoder(backend_out_dim, backbone_dim, base_channels=spunet_base_channels, channels=spunet_channels, layers=spunet_layers, sparse_shape_padding=spunet_sparse_shape_padding)
        self.decoder = VoxelGSAdapter(backend_out_dim=backbone_dim, backbone_dim=backbone_dim, encoder_type='preencoded', **decoder_options)
        self.densification_score_heads = nn.ModuleList()
        if self.densification_score_enabled:
            if densification_score_hidden_dim <= 0:
                raise ValueError('densification_score_hidden_dim must be positive')
            self.densification_score_heads.extend([SparseDensificationScoreHead(backbone_dim, densification_score_hidden_dim) for _ in range(2)])

    def _validate_methods(self, methods):
        methods = tuple(methods)
        if len(methods) > 2 or any((method not in {'point_count', 'learned_score'} for method in methods)):
            raise ValueError('Use at most two point_count/learned_score rounds')
        if 'learned_score' in methods and (not self.densification_score_enabled):
            raise ValueError('learned_score requires densification_score_enabled and a trained score checkpoint')
        return methods

    def backend_voxel_sizes(self, output):
        if self.manual_metric_divisor is None:
            return list(self.multiscale_voxel_sizes)
        intr = output['intrinsics'].detach().float()
        scale = (0.5 * (intr[..., 0, 0] + intr[..., 1, 1])).mean(1) / self.manual_metric_divisor
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid scene scale for metric voxel sizes')
        return [(size / scale)[:, None] for size in self.multiscale_voxel_sizes]

    def forward(self, output, last_backend_voxel_detail=None):
        if last_backend_voxel_detail is None:
            return output
        pyramid = last_backend_voxel_detail['voxel_pyramid']
        batch_size = output['depth'].shape[0]
        encoded = [[] for _ in pyramid]
        counts = [[] for _ in pyramid]
        for batch in range(batch_size):
            features = [d['voxel_feat'][batch] for d in pyramid]
            coords = [d['voxel_grid_coords'][batch] for d in pyramid]
            for (level, feature) in enumerate(self.encoder(features, coords)):
                encoded[level].append(feature)
            points = pyramid[0]['octree_points'][batch]
            views = pyramid[0]['octree_point_view_ids'][batch]
            for (level, detail) in enumerate(pyramid):
                size = detail['voxel_size']
                if torch.is_tensor(size) and size.ndim == 2:
                    size = size[batch].reshape(())
                point_coords = points_to_voxel_grid(points, detail['voxel_grid_origins'][batch], size).floor().long()
                rows = lookup_rows(coords[level], point_coords)
                counts[level].append(max_view_counts(rows, views, len(coords[level])))
        (maps, confidence) = ([], [])
        for (level, detail) in enumerate(pyramid):
            decoded_detail = dict(detail)
            decoded_detail['voxel_feat'] = encoded[level]
            decoded_detail['gaussian_size_ratio'] = detail['voxel_size'] / self.multiscale_voxel_sizes[-1]
            decoded = self.decoder(dict(output), decoded_detail)
            maps.append(decoded['gaussians'])
            confidence.append(decoded.get('gaussian_voxel_depth_conf'))
        if self.densification_score_enabled:
            output['octree_densification_scores'] = [[head(feature.detach() if self.densification_score_detach_features else feature, pyramid[level]['voxel_grid_coords'][batch]) for (batch, feature) in enumerate(encoded[level])] for (level, head) in enumerate(self.densification_score_heads)]
        output['gaussians'] = maps[-1]
        output['gaussian_voxel_depth_conf'] = confidence[-1]
        output['gaussian_pyramid'] = maps
        output['gaussian_pyramid_confidence'] = confidence
        output['octree_metadata'] = {'coords': [d['voxel_grid_coords'] for d in pyramid], 'counts': counts}
        return output

    def prepare_inference(
        self,
        output,
        images,
        valid_mask=None,
        mode=None,
        scene_scale=None,
        methods=None,
    ):
        mode = self.inference_mode if mode is None else mode
        if mode not in {'adaptive', 'finest', 'all'}:
            raise ValueError('octdensify_mode must be adaptive, finest or all')
        methods = self.octree_densify_methods if methods is None else self._validate_methods(methods)
        fractions = split_fractions(self.octree_split_fraction, len(methods))
        if mode == 'all':
            return
        maps = output.pop('gaussian_pyramid')
        confidence = output.pop('gaussian_pyramid_confidence')
        metadata = output.pop('octree_metadata')
        if mode == 'finest':
            return
        (scenes, packed_confidence, selected_coords, selected_levels) = ([], [], [], [])
        splits = self.decoder.gaussian_splits
        for batch in range(images.shape[0]):
            coords = [c[batch] for c in metadata['coords']]
            counts = [c[batch] for c in metadata['counts']]

            def score(selected, level, method=None):
                method = methods[level] if method is None else method
                if method == 'learned_score':
                    return output['octree_densification_scores'][level][batch][selected[level]]
                raise ValueError('Only learned_score and point_count are supported')
            selected = select_leaves(coords, counts, methods, fractions, score, strategy=self.octree_split_strategy)
            scenes.append(gather_gaussians(maps, batch, selected, splits))
            selected_coords.append(torch.cat([c[r] for (c, r) in zip(coords, selected)]))
            selected_levels.append(torch.cat([r.new_full((len(r),), level) for (level, r) in enumerate(selected)]))
            if confidence[0] is not None:
                parts = [c[batch, (r[:, None] * splits + torch.arange(splits, device=r.device)).reshape(-1)] for (c, r) in zip(confidence, selected)]
                packed_confidence.append(torch.cat(parts))
        output['gaussians'] = pack_scenes(scenes)
        output['gaussian_voxel_depth_conf'] = None
        if packed_confidence:
            count = output['gaussians'].means.shape[1]
            output['gaussian_voxel_depth_conf'] = torch.stack([torch.cat((c, c.new_zeros(count - len(c)))) for c in packed_confidence])
        output['octree_leaf_coords'] = selected_coords
        output['octree_leaf_levels'] = selected_levels
