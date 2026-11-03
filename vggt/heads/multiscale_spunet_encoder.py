"""Sparse U-Net over the explicit 8/4/2 mm occupied voxel maps."""
from __future__ import annotations

import torch
from torch import nn
import spconv.pytorch as spconv

from vggt.heads.spunet_encoder import SpUNetVoxelEncoder


def lookup_rows(source, query):
    """Return source row indices for integer coordinate queries (no hash collisions)."""
    if not len(query):
        return query.new_empty((0,), dtype=torch.long)
    if not len(source):
        raise ValueError("Cannot find children in an empty parent map")
    extent = torch.maximum(source.amax(0), query.amax(0)).long() + 1
    def keys(coords):
        return (coords[:, 0].long() * extent[1] + coords[:, 1]) * extent[2] + coords[:, 2]
    sorted_keys, order = keys(source).sort()
    query_keys = keys(query)
    positions = torch.searchsorted(sorted_keys, query_keys)
    if (positions >= len(source)).any() or not torch.equal(sorted_keys[positions], query_keys):
        raise ValueError("Voxel pyramid has missing parents or inconsistent grid origins")
    return order[positions]


class SparseLayerNorm(nn.LayerNorm):
    """Normalize in FP32, then preserve the sparse convolution feature dtype."""

    def forward(self, features):
        return super().forward(features.float()).to(dtype=features.dtype)


class MultiscaleSpUNetVoxelEncoder(nn.Module):
    """SpUNet-depth encoder/decoder with three explicit input/output voxel maps.

    The default topology has five resolutions (2/4/8/16/32 mm), with the same
    four encoder and four decoder widths/block counts as the original SpUNet.
    Only the finest three levels receive point-aggregated descriptors and emit
    Gaussian features. Deeper levels supply spatial context to all three maps.
    LayerNorm supports even a single occupied voxel at the deepest level.
    """

    def __init__(
        self, in_channels, out_channels, base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2), sparse_shape_padding=96,
    ):
        super().__init__()
        self.sparse_conv_fp16 = False
        channels, layers = tuple(int(v) for v in channels), tuple(int(v) for v in layers)
        if len(channels) != len(layers) or len(channels) < 4 or len(channels) % 2:
            raise ValueError("channels/layers need matching even lengths and at least two down/up stages")
        if any(v <= 0 for v in (base_channels, sparse_shape_padding, *channels, *layers)):
            raise ValueError("Channel widths, block counts and sparse padding must be positive")
        self.out_channels = out_channels
        self.num_stages = len(channels) // 2
        self.map_levels = 3
        self.sparse_shape_padding = int(sparse_shape_padding)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, 5, padding=1,
                             bias=False, indice_key="pyramid_stem"),
            SparseLayerNorm(base_channels), nn.ReLU(),
        )
        # The 4/8 mm point descriptors are injected after the first two downsamples.
        self.map_projections = nn.ModuleList(nn.Linear(in_channels, channels[i]) for i in range(2))
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        encoder_widths = (base_channels, *channels[:self.num_stages])
        for stage in range(self.num_stages):
            width = channels[stage]
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(encoder_widths[stage], width, 2, stride=2,
                                    bias=False, indice_key=f"pyramid_down{stage}"),
                SparseLayerNorm(width), nn.ReLU(),
            ))
            self.enc.append(SpUNetVoxelEncoder._make_blocks(
                layers[stage], 2 * width if stage < 2 else width, width,
                SparseLayerNorm, f"pyramid_enc{stage + 1}",
            ))
            decoder_width = channels[-stage - 1]
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(channels[-stage - 2], decoder_width, 2,
                                           bias=False, indice_key=f"pyramid_down{stage}"),
                SparseLayerNorm(decoder_width), nn.ReLU(),
            ))
            self.dec.append(SpUNetVoxelEncoder._make_blocks(
                layers[-stage - 1], decoder_width + encoder_widths[stage], decoder_width,
                SparseLayerNorm, f"pyramid_dec{stage}",
            ))
        output_widths = [encoder_widths[-1] if level == self.num_stages else channels[-level - 1]
                         for level in range(self.map_levels)]
        self.heads = nn.ModuleList(nn.Linear(width, out_channels) for width in output_widths)

    def forward(self, features, coords):
        if len(features) != self.map_levels or len(coords) != self.map_levels:
            raise ValueError("Provide three occupied input maps, coarse to fine")
        # spconv supports FP16 kernels; keep BF16 outside this precision boundary.
        enabled = self.sparse_conv_fp16 and features[0].is_cuda
        with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
            outputs = self._forward(features, coords, enabled)
        return [output.float() for output in outputs]

    def _forward(self, features, coords, use_fp16=False):
        if len(features) != self.map_levels or len(coords) != self.map_levels:
            raise ValueError("Provide three occupied input maps, coarse to fine")
        # Public ordering is coarse to fine; internal U-Net ordering is fine to coarse.
        features, coords = list(reversed(features)), list(reversed(coords))
        if not len(coords[0]):
            return [f.new_zeros((0, self.out_channels)) for f in reversed(features)]
        # Keep the spatial bounds divisible by every stride so boundary cells
        # survive all four downsampling steps, including very small scenes.
        stride = 2 ** self.num_stages
        shape = (((coords[0].amax(0).long() + self.sparse_shape_padding + stride - 1)
                  // stride) * stride).tolist()
        indices = torch.cat((torch.zeros_like(coords[0][:, :1]), coords[0]), 1).int()
        sparse = spconv.SparseConvTensor(
            features[0].to(dtype=torch.float16 if use_fp16 else torch.float32),
            indices, shape, 1)
        sparse = self.conv_input(sparse)
        skips = [sparse]
        for stage in range(self.num_stages):
            sparse = self.down[stage](sparse)
            level = stage + 1
            if level < self.map_levels:
                rows = lookup_rows(coords[level], sparse.indices[:, 1:].long())
                raw = self.map_projections[stage](features[level].float())[rows]
                sparse = sparse.replace_feature(torch.cat((sparse.features, raw), -1))
            sparse = self.enc[stage](sparse)
            skips.append(sparse)
        outputs = [None] * self.map_levels
        for level in reversed(range(self.num_stages + 1)):
            if level < self.num_stages:
                sparse = self.up[level](sparse)
                skip = skips[level]
                rows = lookup_rows(skip.indices[:, 1:].long(), sparse.indices[:, 1:].long())
                sparse = self.dec[level](sparse.replace_feature(
                    torch.cat((sparse.features, skip.features[rows]), -1)))
            if level < self.map_levels:
                rows = lookup_rows(sparse.indices[:, 1:].long(), coords[level])
                outputs[level] = self.heads[level](sparse.features)[rows]
        return list(reversed(outputs))
