"""
roi_align.py  –  Quantization-free RoI feature extraction.
Paper §3  RoIAlign

Wraps torchvision's roi_align with the multi-level FPN assignment
logic described in §3 / Appendix B of the paper.
"""

import math
import torch
import torch.nn as nn
from torchvision.ops import roi_align


def _assign_fpn_level(box_areas, k0=4, canonical_area=224**2,
                      min_level=2, max_level=5):
    """
    Eq. from FPN paper: k = floor(k0 + log2(sqrt(area) / 224))
    Clipped to [min_level, max_level].
    """
    target = torch.floor(k0 + torch.log2(
        torch.sqrt(box_areas) / math.sqrt(canonical_area)
    )).long()
    return target.clamp(min_level, max_level)


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoI Align for FPN feature maps (P2 … P5).

    Args
    ----
    output_size : int or (int, int)   –  pooled output H × W
    sampling_ratio : int              –  bilinear sampling points per bin
    """

    def __init__(self, output_size=7, sampling_ratio=2):
        super().__init__()
        self.output_size    = output_size if isinstance(output_size, (list, tuple)) \
                              else (output_size, output_size)
        self.sampling_ratio = sampling_ratio

    def forward(self, feature_maps, boxes, image_size):
        """
        Parameters
        ----------
        feature_maps : list[Tensor]   P2…P5, each (N, 256, Hi, Wi)
        boxes        : list[Tensor]   per-image boxes  (K_i, 4) in xyxy image coords
        image_size   : (H, W)

        Returns
        -------
        Tensor  (total_boxes, 256, out_H, out_W)
        """
        # Build flat roi list with batch index prefix: (img_idx, x1, y1, x2, y2)
        rois = []
        for img_idx, b in enumerate(boxes):
            if b.numel() == 0:
                continue
            idx_col = b.new_full((b.shape[0], 1), img_idx)
            rois.append(torch.cat([idx_col, b], dim=1))
        if not rois:
            dummy = feature_maps[0].new_zeros(
                0, feature_maps[0].shape[1], *self.output_size
            )
            return dummy
        rois = torch.cat(rois, dim=0)   # (R, 5)

        # Area for level assignment (use raw boxes, no index col)
        xy1  = rois[:, 1:3]
        xy2  = rois[:, 3:5]
        wh   = (xy2 - xy1).clamp(min=1e-5)
        areas = wh[:, 0] * wh[:, 1]
        levels = _assign_fpn_level(areas)   # values in {2,3,4,5}

        H, W = image_size
        results  = torch.zeros(
            rois.shape[0], feature_maps[0].shape[1], *self.output_size,
            device=feature_maps[0].device
        )

        for lvl_idx, lvl in enumerate(range(2, 6)):          # P2→P5
            mask  = (levels == lvl).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            lvl_rois   = rois[mask]
            spatial_scale = feature_maps[lvl_idx].shape[-1] / W
            pooled = roi_align(
                feature_maps[lvl_idx],
                lvl_rois,
                output_size=self.output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True,           # removes the 0.5 offset = true RoIAlign
            )
            results[mask] = pooled

        return results