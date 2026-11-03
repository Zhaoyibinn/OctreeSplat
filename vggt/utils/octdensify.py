"""Occupied parent/child selection and GGGS position-gradient priorities."""
from __future__ import annotations
import math
from dataclasses import fields
from numbers import Real
import torch
from vggt.models.depth_anything_3.specs import Gaussians

def split_fractions(value, rounds):
    result = (float(value),) * rounds if isinstance(value, Real) else tuple((float(v) for v in value))
    if len(result) != rounds or any((not math.isfinite(v) or not 0 <= v <= 1 for v in result)):
        raise ValueError('One finite split fraction in [0, 1] is required per round')
    return result

def max_view_counts(point_leaf, view_ids, count):
    """Match svrvggt_train: maximum points from any one view, not sum of views."""
    result = torch.zeros(count, dtype=torch.long, device=point_leaf.device)
    if len(point_leaf):
        num_views = int(view_ids.max()) + 1
        (pairs, counts) = torch.unique(point_leaf * num_views + view_ids, return_counts=True)
        result.scatter_reduce_(0, pairs // num_views, counts, reduce='amax', include_self=True)
    return result

def choose_splits(scores, fraction):
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite octdensify priorities')
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected[scores.argsort(descending=True, stable=True)[:math.ceil(len(scores) * fraction)]] = True
    return selected

def gather_gaussians(maps, batch_idx, selections, splits):
    """Pack selected voxel rows into one scene, preserving the voxel/splat order."""
    attrs = {}
    for field in fields(Gaussians):
        if getattr(maps[0], field.name) is None:
            attrs[field.name] = None
            continue
        parts = []
        for (gs, rows) in zip(maps, selections):
            indices = (rows[:, None] * splits + torch.arange(splits, device=rows.device)).reshape(-1)
            parts.append(getattr(gs, field.name)[batch_idx, indices])
        attrs[field.name] = torch.cat(parts, 0).unsqueeze(0)
    return Gaussians(**attrs)

def pack_scenes(scenes):
    count = max(1, max((gs.means.shape[1] for gs in scenes)))
    attrs = {}
    for field in fields(Gaussians):
        if getattr(scenes[0], field.name) is None:
            attrs[field.name] = None
            continue
        parts = []
        for scene in scenes:
            value = getattr(scene, field.name)[0]
            pad = value.new_zeros((count - len(value), *value.shape[1:]))
            parts.append(torch.cat((value, pad)))
        attrs[field.name] = torch.stack(parts)
    return Gaussians(**attrs)

def select_leaves(coords, counts, methods, fractions, score_fn, strategy='per_level'):
    """Replace selected parents with occupied children.

    per_level refines 0->1, then 1->2. mixed ranks all currently splittable
    leaves together using the current round's method and fraction. Newly
    introduced children cannot split again in the same round. Score callbacks
    in mixed mode receive the current round's method as a keyword argument.
    """
    from vggt.heads.multiscale_spunet_encoder import lookup_rows
    if strategy not in {'per_level', 'mixed'}:
        raise ValueError('octree_split_strategy must be per_level or mixed')
    selected = [torch.arange(len(coords[0]), device=coords[0].device), coords[0].new_empty(0), coords[0].new_empty(0)]
    for (level, (method, fraction)) in enumerate(zip(methods, fractions)):
        if strategy == 'mixed':
            if fraction == 0:
                continue
            candidate_levels = range(level + 1)
            priorities = []
            for candidate_level in candidate_levels:
                rows = selected[candidate_level]
                if not len(rows):
                    priorities.append(counts[candidate_level][rows].float())
                    continue
                scores = counts[candidate_level][rows] if method == 'point_count' else score_fn(selected, candidate_level, method=method)
                if scores.shape != rows.shape:
                    raise ValueError('One priority is required per candidate parent')
                priorities.append(scores)
            if not any((len(s) for s in priorities)):
                continue
            split = choose_splits(torch.cat(priorities), fraction)
            next_selected = list(selected)
            children = {}
            offset = 0
            for candidate_level in candidate_levels:
                rows = selected[candidate_level]
                mask = split[offset:offset + len(rows)]
                offset += len(rows)
                next_selected[candidate_level] = rows[~mask]
                if not mask.any():
                    continue
                parents = lookup_rows(coords[candidate_level], coords[candidate_level + 1] // 2)
                split_mask = torch.zeros(len(coords[candidate_level]), device=rows.device, dtype=torch.bool)
                split_mask[rows[mask]] = True
                children[candidate_level + 1] = split_mask[parents].nonzero().flatten()
            for (child_level, rows) in children.items():
                next_selected[child_level] = torch.cat((next_selected[child_level], rows)).sort().values
            selected = next_selected
            continue
        rows = selected[level]
        if not len(rows) or fraction == 0:
            continue
        scores = counts[level][rows] if method == 'point_count' else score_fn(selected, level)
        if scores.shape != rows.shape:
            raise ValueError('One priority is required per candidate parent')
        split = choose_splits(scores, fraction)
        parents = lookup_rows(coords[level], coords[level + 1] // 2)
        split_mask = torch.zeros(len(coords[level]), device=rows.device, dtype=torch.bool)
        split_mask[rows[split]] = True
        selected[level] = rows[~split]
        selected[level + 1] = split_mask[parents].nonzero().flatten()
    return selected
