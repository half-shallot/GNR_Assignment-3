"""
rpn.py  –  Region Proposal Network
Paper §3  Faster R-CNN / Mask R-CNN

Generates anchors, predicts objectness + delta offsets,
applies NMS, and returns top-K proposals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image
import math


# ---------------------------------------------------------------------------
# Anchor Generator
# ---------------------------------------------------------------------------

class AnchorGenerator(nn.Module):
    """
    5 scales × 3 aspect ratios anchors at each spatial location,
    one set per FPN level  (§3.1 Implementation Details).
    """

    def __init__(
        self,
        sizes       = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5,
    ):
        super().__init__()
        self.sizes         = sizes
        self.aspect_ratios = aspect_ratios

    @torch.no_grad()
    def _base_anchors(self, sizes, ratios, device):
        sizes  = torch.as_tensor(sizes,  dtype=torch.float32, device=device)
        ratios = torch.as_tensor(ratios, dtype=torch.float32, device=device)
        h_ratios = torch.sqrt(ratios)
        w_ratios = 1.0 / h_ratios
        ws = (w_ratios[:, None] * sizes[None, :]).view(-1)
        hs = (h_ratios[:, None] * sizes[None, :]).view(-1)
        return torch.stack([-ws, -hs, ws, hs], dim=1) / 2.0

    @torch.no_grad()
    def forward(self, feature_maps, image_size):
        """
        Returns list of anchor tensors, one per FPN level.
        Each tensor: (Hi * Wi * A, 4) in xyxy image coords.
        """
        H_img, W_img = image_size
        all_anchors  = []

        for lvl, (fmap, sizes, ratios) in enumerate(
            zip(feature_maps, self.sizes, self.aspect_ratios)
        ):
            _, _, H, W = fmap.shape
            stride_h   = H_img / H
            stride_w   = W_img / W

            base = self._base_anchors(sizes, ratios, fmap.device)  # (A, 4)

            shift_x = (torch.arange(W, device=fmap.device) + 0.5) * stride_w
            shift_y = (torch.arange(H, device=fmap.device) + 0.5) * stride_h
            sy, sx  = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts  = torch.stack([sx.ravel(), sy.ravel(),
                                   sx.ravel(), sy.ravel()], dim=1)  # (HW, 4)

            anchors = (shifts[:, None, :] + base[None, :, :]).reshape(-1, 4)
            all_anchors.append(anchors)

        return all_anchors   # list[Tensor(HiWiA, 4)]


# ---------------------------------------------------------------------------
# RPN Head
# ---------------------------------------------------------------------------

class RPNHead(nn.Module):
    """Shared 3×3 conv → objectness logits + bbox deltas."""

    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv      = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, 1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        logits, deltas = [], []
        for x in features:
            t = F.relu(self.conv(x))
            logits.append(self.cls_logits(t))
            deltas.append(self.bbox_pred(t))
        return logits, deltas


# ---------------------------------------------------------------------------
# Box coding helpers
# ---------------------------------------------------------------------------

def decode_boxes(anchors, deltas, weights=(1., 1., 1., 1.)):
    """Apply delta offsets (tx,ty,tw,th) to anchors → predicted boxes."""
    wx, wy, ww, wh = weights
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = deltas[..., 2] / ww
    dh = deltas[..., 3] / wh

    dw = dw.clamp(max=math.log(1000. / 16))
    dh = dh.clamp(max=math.log(1000. / 16))

    # Anchor centres + sizes
    wa = anchors[..., 2] - anchors[..., 0]
    ha = anchors[..., 3] - anchors[..., 1]
    cx = anchors[..., 0] + 0.5 * wa
    cy = anchors[..., 1] + 0.5 * ha

    px = dx * wa + cx
    py = dy * ha + cy
    pw = torch.exp(dw) * wa
    ph = torch.exp(dh) * ha

    return torch.stack([px - pw/2, py - ph/2,
                        px + pw/2, py + ph/2], dim=-1)


def encode_boxes(anchors, gt_boxes, weights=(1., 1., 1., 1.)):
    """Ground-truth → delta targets."""
    wx, wy, ww, wh = weights
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * wa
    cy = anchors[:, 1] + 0.5 * ha

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gcx= gt_boxes[:, 0] + 0.5 * gw
    gcy= gt_boxes[:, 1] + 0.5 * gh

    tx = wx * (gcx - cx) / wa
    ty = wy * (gcy - cy) / ha
    tw = ww * torch.log(gw / wa + 1e-8)
    th = wh * torch.log(gh / ha + 1e-8)
    return torch.stack([tx, ty, tw, th], dim=1)


# ---------------------------------------------------------------------------
# Full RPN Module
# ---------------------------------------------------------------------------

class RPN(nn.Module):
    """
    RPN that works on the 5 FPN levels (P2–P6).

    During training also computes classification + regression losses.
    During inference returns top-K proposals after NMS.
    """

    def __init__(
        self,
        anchor_generator      : AnchorGenerator,
        head                  : RPNHead,
        # Proposal cfg
        pre_nms_top_n_train   = 2000,
        pre_nms_top_n_test    = 1000,
        post_nms_top_n_train  = 2000,
        post_nms_top_n_test   = 1000,
        nms_thresh            = 0.7,
        min_size              = 0.,
        # Training cfg
        fg_iou_thresh         = 0.7,
        bg_iou_thresh         = 0.3,
        batch_size_per_image  = 256,
        positive_fraction     = 0.5,
    ):
        super().__init__()
        self.anchor_generator     = anchor_generator
        self.head                 = head
        self.pre_nms_top_n_train  = pre_nms_top_n_train
        self.pre_nms_top_n_test   = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test  = post_nms_top_n_test
        self.nms_thresh           = nms_thresh
        self.min_size             = min_size
        self.fg_iou_thresh        = fg_iou_thresh
        self.bg_iou_thresh        = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction    = positive_fraction

    # ------------------------------------------------------------------
    def _filter_proposals(self, proposals, scores, image_size, is_training):
        pre_n  = self.pre_nms_top_n_train  if is_training else self.pre_nms_top_n_test
        post_n = self.post_nms_top_n_train if is_training else self.post_nms_top_n_test

        # clip
        proposals = clip_boxes_to_image(proposals, image_size)

        # remove tiny
        if self.min_size > 0:
            wh   = proposals[:, 2:] - proposals[:, :2]
            keep = (wh[:, 0] >= self.min_size) & (wh[:, 1] >= self.min_size)
            proposals, scores = proposals[keep], scores[keep]

        # top-k before NMS
        if pre_n < scores.shape[0]:
            topk = scores.topk(pre_n).indices
            proposals, scores = proposals[topk], scores[topk]

        keep = batched_nms(proposals, scores,
                           torch.zeros_like(scores, dtype=torch.long),
                           self.nms_thresh)
        keep = keep[:post_n]
        return proposals[keep], scores[keep]

    # ------------------------------------------------------------------
    def _assign_targets(self, anchors, gt_boxes):
        """Returns (labels, matched_gt_boxes) for one image."""
        if gt_boxes.numel() == 0:
            labels    = torch.zeros(anchors.shape[0], dtype=torch.long,
                                    device=anchors.device)
            matched   = torch.zeros_like(anchors)
            return labels, matched

        iou = box_iou(anchors, gt_boxes)            # (A, G)
        matched_vals, matched_idx = iou.max(dim=1)  # (A,)

        labels = torch.full((anchors.shape[0],), -1,
                            dtype=torch.long, device=anchors.device)
        labels[matched_vals >= self.fg_iou_thresh] =  1
        labels[matched_vals <  self.bg_iou_thresh] =  0

        # ensure each GT is matched to at least one anchor
        best_anchor_per_gt = iou.argmax(dim=0)
        labels[best_anchor_per_gt] = 1

        # subsample
        n_pos = int(self.batch_size_per_image * self.positive_fraction)
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        if pos_idx.numel() > n_pos:
            disable = pos_idx[torch.randperm(pos_idx.numel(),
                                             device=anchors.device)[n_pos:]]
            labels[disable] = -1

        n_neg = self.batch_size_per_image - (labels == 1).sum().item()
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        if neg_idx.numel() > n_neg:
            disable = neg_idx[torch.randperm(neg_idx.numel(),
                                             device=anchors.device)[int(n_neg):]]
            labels[disable] = -1

        matched_boxes = gt_boxes[matched_idx]
        return labels, matched_boxes

    # ------------------------------------------------------------------
    def forward(self, features, image_size, targets=None):
        """
        features : list of FPN feature maps [P2..P6]
        image_size: (H, W)
        targets  : list of dicts with 'boxes' (for training)

        Returns
        -------
        proposals : list[Tensor]   (K_i, 4) per image
        rpn_losses: dict  (empty during inference)
        """
        logits_list, deltas_list = self.head(features)
        anchors_per_lvl          = self.anchor_generator(features, image_size)

        # Flatten all levels together
        # logits_list: [(N, A, H, W), ...]
        N = logits_list[0].shape[0]

        # Per-image proposals
        all_proposals = []

        for img_idx in range(N):
            lvl_anchors, lvl_scores, lvl_deltas = [], [], []
            for lvl_idx in range(len(features)):
                a = anchors_per_lvl[lvl_idx]                   # (HWA, 4)
                s = logits_list[lvl_idx][img_idx]              # (A, H, W)
                d = deltas_list[lvl_idx][img_idx]              # (4A, H, W)

                A     = s.shape[0]
                H, W  = s.shape[1], s.shape[2]
                s     = s.permute(1, 2, 0).reshape(-1)         # (HWA,)
                d     = d.permute(1, 2, 0).reshape(-1, 4)      # (HWA, 4)

                props = decode_boxes(a, d)
                lvl_anchors.append(a)
                lvl_scores.append(torch.sigmoid(s))
                lvl_deltas.append(d)

            cat_scores = torch.cat(lvl_scores, dim=0)
            cat_props  = torch.cat([decode_boxes(a, d) for a, d in
                                    zip(lvl_anchors, lvl_deltas)], dim=0)

            props, _ = self._filter_proposals(
                cat_props, cat_scores, image_size,
                is_training=self.training
            )
            all_proposals.append(props)

        # ---- Training losses ----
        rpn_losses = {}
        if self.training and targets is not None:
            cls_losses, reg_losses = [], []
            all_anchors = torch.cat(anchors_per_lvl, dim=0)  # (TotalA, 4)

            for img_idx, tgt in enumerate(targets):
                gt_boxes = tgt["boxes"].to(all_anchors.device)
                labels, matched_gt = self._assign_targets(all_anchors, gt_boxes)

                # Objectness loss (binary cross-entropy on sampled set)
                sampled = (labels >= 0).nonzero(as_tuple=True)[0]
                flat_logits = torch.cat(
                    [l[img_idx].permute(1, 2, 0).reshape(-1)
                     for l in logits_list], dim=0
                )
                cls_loss = F.binary_cross_entropy_with_logits(
                    flat_logits[sampled],
                    labels[sampled].float()
                )
                cls_losses.append(cls_loss)

                # Box regression loss (Smooth-L1 on positives only)
                pos     = (labels == 1).nonzero(as_tuple=True)[0]
                if pos.numel() > 0:
                    flat_deltas = torch.cat(
                        [d[img_idx].permute(1, 2, 0).reshape(-1, 4)
                         for d in deltas_list], dim=0
                    )
                    tgt_deltas  = encode_boxes(all_anchors[pos], matched_gt[pos])
                    reg_loss = F.smooth_l1_loss(
                        flat_deltas[pos], tgt_deltas, beta=1.0 / 9
                    )
                    reg_losses.append(reg_loss)

            rpn_losses["rpn_cls_loss"] = torch.stack(cls_losses).mean()
            if reg_losses:
                rpn_losses["rpn_box_loss"] = torch.stack(reg_losses).mean()
            else:
                rpn_losses["rpn_box_loss"] = all_anchors.new_zeros(1).squeeze()

        return all_proposals, rpn_losses