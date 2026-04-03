"""
roi_head.py  –  RoI classification, bbox-regression & mask prediction heads.
Paper §3  Mask R-CNN  /  Figure 4  Head Architecture

Multi-task loss:  L = L_cls + L_box + L_mask   (Eq. 1 in the paper)
Mask branch      : small FCN  →  K binary masks of resolution 28×28
Decoupled masks  : per-pixel sigmoid + binary cross-entropy  (§3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, clip_boxes_to_image, batched_nms

from rpn import encode_boxes, decode_boxes


# ---------------------------------------------------------------------------
# Box / Classification Head  (FPN version, right panel of Figure 4)
# ---------------------------------------------------------------------------

class BoxHead(nn.Module):
    """
    Two FC layers: 7×7×256 → 1024 → 1024
    Then separate branches for class scores and box deltas.
    """

    def __init__(self, in_channels=256, roi_size=7, num_classes=81):
        super().__init__()
        flatten = in_channels * roi_size * roi_size
        self.fc1 = nn.Linear(flatten, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred  = nn.Linear(1024, num_classes * 4)

        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight,  std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)


# ---------------------------------------------------------------------------
# Mask Head  (§3  Mask Representation  +  Figure 4 right panel)
# ---------------------------------------------------------------------------

class MaskHead(nn.Module):
    """
    4 × conv(256, 3×3) → deconv(256, 2×2, s=2) → conv(K, 1×1)
    Output: (R, K, 28, 28) binary mask logits, one per class.
    """

    def __init__(self, in_channels=256, num_classes=81):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1),
                       nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
                   nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels, num_classes, 1)]
        self.convs = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.convs(x)   # (R, K, 28, 28)


# ---------------------------------------------------------------------------
# Proposal Sampler  (training only)
# ---------------------------------------------------------------------------

class ProposalSampler:
    """
    Matches RoI proposals to GT, samples 512 per image (1:3 pos:neg ratio).
    """

    def __init__(self, fg_iou=0.5, bg_iou_hi=0.5, bg_iou_lo=0.0,
                 batch_size=512, pos_fraction=0.25):
        self.fg_iou      = fg_iou
        self.bg_iou_hi   = bg_iou_hi
        self.bg_iou_lo   = bg_iou_lo
        self.batch_size  = batch_size
        self.pos_fraction= pos_fraction

    def __call__(self, proposals, targets):
        """
        Returns
        -------
        sampled_props  : list[Tensor (K, 4)]
        sampled_labels : list[Tensor (K,)]   0 = background, 1..C = class
        matched_gt_boxes: list[Tensor (K, 4)]
        matched_gt_masks: list[Tensor | None]
        """
        out_props, out_labels, out_boxes, out_masks = [], [], [], []

        for props, tgt in zip(proposals, targets):
            gt_boxes  = tgt["boxes"]
            gt_labels = tgt["labels"]
            gt_masks  = tgt.get("masks", None)   # (G, H, W) bool or None

            if gt_boxes.numel() == 0:
                n = min(self.batch_size, props.shape[0])
                out_props.append(props[:n])
                out_labels.append(torch.zeros(n, dtype=torch.long,
                                              device=props.device))
                out_boxes.append(props[:n])
                out_masks.append(None)
                continue

            # Combine GT boxes with proposals (GT always included)
            all_props = torch.cat([props, gt_boxes], dim=0)
            iou       = box_iou(all_props, gt_boxes)          # (R, G)
            best_iou, matched_idx = iou.max(dim=1)

            labels    = gt_labels[matched_idx]
            labels[best_iou < self.bg_iou_hi] = 0            # bg if below threshold
            labels[best_iou < self.bg_iou_lo] = -1           # ignore

            # Subsample
            n_pos = int(self.batch_size * self.pos_fraction)
            pos   = (labels > 0).nonzero(as_tuple=True)[0]
            neg   = (labels == 0).nonzero(as_tuple=True)[0]

            pos = pos[torch.randperm(pos.numel(), device=props.device)[:n_pos]]
            n_neg = min(self.batch_size - pos.numel(), neg.numel())
            neg = neg[torch.randperm(neg.numel(), device=props.device)[:n_neg]]

            keep = torch.cat([pos, neg])
            out_props.append(all_props[keep])
            out_labels.append(labels[keep])
            out_boxes.append(gt_boxes[matched_idx[keep]])

            if gt_masks is not None:
                out_masks.append(gt_masks[matched_idx[keep]])
            else:
                out_masks.append(None)

        return out_props, out_labels, out_boxes, out_masks


# ---------------------------------------------------------------------------
# Full RoI Head
# ---------------------------------------------------------------------------

class RoIHead(nn.Module):
    """
    Wraps BoxHead + MaskHead, handles sampling during training and
    produces detection outputs during inference.
    """

    def __init__(
        self,
        box_roi_pool,     # MultiScaleRoIAlign(output_size=7)
        mask_roi_pool,    # MultiScaleRoIAlign(output_size=14)
        box_head,         # BoxHead
        mask_head,        # MaskHead
        num_classes = 81,
        score_thresh= 0.05,
        nms_thresh  = 0.5,
        detections_per_img = 100,
        mask_thresh = 0.5,
    ):
        super().__init__()
        self.box_roi_pool  = box_roi_pool
        self.mask_roi_pool = mask_roi_pool
        self.box_head      = box_head
        self.mask_head     = mask_head
        self.num_classes   = num_classes
        self.score_thresh  = score_thresh
        self.nms_thresh    = nms_thresh
        self.detections_per_img = detections_per_img
        self.mask_thresh   = mask_thresh
        self.sampler       = ProposalSampler()

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _cls_box_loss(self, cls_scores, bbox_pred, labels, gt_boxes, proposals):
        """Equation 1 L_cls + L_box."""
        cls_loss = F.cross_entropy(cls_scores, labels)

        pos_mask = labels > 0
        if pos_mask.sum() == 0:
            box_loss = bbox_pred.sum() * 0
        else:
            pos_labels   = labels[pos_mask]
            pos_pred     = bbox_pred[pos_mask]                    # (P, C*4)
            pos_pred     = pos_pred.reshape(-1, self.num_classes, 4)
            pos_pred     = pos_pred[torch.arange(pos_pred.shape[0]), pos_labels]  # (P,4)

            pos_props    = proposals[pos_mask]
            pos_gt       = gt_boxes[pos_mask]
            tgt_deltas   = encode_boxes(pos_props, pos_gt)

            box_loss = F.smooth_l1_loss(pos_pred, tgt_deltas, beta=1.0)

        return cls_loss, box_loss

    @staticmethod
    def _mask_loss(mask_logits, proposals, gt_masks, labels):
        """
        L_mask: binary cross-entropy on the k-th mask for each RoI,
        where k is the GT class label.  (§3 Mask R-CNN  Lmask paragraph)
        """
        pos_mask = labels > 0
        if pos_mask.sum() == 0:
            return mask_logits.sum() * 0

        pos_props  = proposals[pos_mask]
        pos_labels = labels[pos_mask]
        pos_logits = mask_logits[pos_mask]   # (P, K, 28, 28)

        # Select the k-th mask
        pred = pos_logits[torch.arange(pos_logits.shape[0]), pos_labels]  # (P, 28, 28)

        # Crop GT masks to proposal box & resize to 28×28
        if gt_masks is None:
            return pred.sum() * 0

        tgt_masks = []
        for i, (box, m) in enumerate(zip(pos_props, gt_masks[pos_mask])):
            x1, y1, x2, y2 = box.long().clamp(min=0)
            m_crop = m[y1:y2+1, x1:x2+1].float().unsqueeze(0).unsqueeze(0)
            if m_crop.numel() == 0:
                m_crop = m.new_zeros(1, 1, 1, 1).float()
            m_resized = F.interpolate(m_crop, size=(28, 28),
                                      mode="bilinear", align_corners=False)
            tgt_masks.append(m_resized.squeeze())

        tgt = torch.stack(tgt_masks)   # (P, 28, 28)
        return F.binary_cross_entropy_with_logits(pred, tgt)

    # ------------------------------------------------------------------
    # Inference post-processing
    # ------------------------------------------------------------------

    def _post_process(self, cls_scores, bbox_pred, proposals, image_size):
        scores  = F.softmax(cls_scores, dim=-1)  # (R, C)
        C       = self.num_classes

        all_boxes, all_scores, all_labels = [], [], []

        for cls_idx in range(1, C):              # skip background
            s   = scores[:, cls_idx]
            d   = bbox_pred[:, cls_idx*4:(cls_idx+1)*4]
            b   = decode_boxes(proposals, d)
            b   = clip_boxes_to_image(b, image_size)

            keep = s > self.score_thresh
            b, s = b[keep], s[keep]
            all_boxes.append(b)
            all_scores.append(s)
            all_labels.append(torch.full_like(s, cls_idx, dtype=torch.long))

        if not all_boxes:
            empty = proposals.new_zeros(0, 4)
            return empty, empty.new_zeros(0), empty.new_zeros(0, dtype=torch.long)

        boxes  = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        keep   = batched_nms(boxes, scores, labels, self.nms_thresh)
        keep   = keep[:self.detections_per_img]
        return boxes[keep], scores[keep], labels[keep]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features, proposals, image_size, targets=None):
        """
        features  : list of FPN maps  [P2..P5]  (P6 not used in RoI head)
        proposals : list[Tensor (K_i, 4)]
        image_size: (H, W)
        targets   : list[dict] with 'boxes', 'labels', 'masks' (training only)

        Returns
        -------
        detections : list[dict]  (inference)  or []  (training)
        roi_losses : dict
        """
        roi_losses = {}

        if self.training and targets is not None:
            proposals, labels, gt_boxes, gt_masks = \
                self.sampler(proposals, targets)
            # Flatten across images for pooling
            flat_labels   = torch.cat(labels)
            flat_gt_boxes = torch.cat(gt_boxes)
        else:
            flat_labels   = None
            flat_gt_boxes = None

        # ---- Pool box features ----
        fpn4 = features[:4]   # P2..P5 only
        box_features = self.box_roi_pool(fpn4, proposals, image_size)
        cls_scores, bbox_pred = self.box_head(box_features)

        if self.training and flat_labels is not None:
            flat_proposals = torch.cat(proposals)
            cls_loss, box_loss = self._cls_box_loss(
                cls_scores, bbox_pred, flat_labels, flat_gt_boxes, flat_proposals
            )
            roi_losses["roi_cls_loss"] = cls_loss
            roi_losses["roi_box_loss"] = box_loss

            # ---- Mask branch  (positive RoIs only) ----
            pos_mask = flat_labels > 0
            if pos_mask.sum() > 0:
                mask_losses = []
                for props_i, labels_i, gt_masks_i in zip(proposals, labels, gt_masks):
                    pos_i = labels_i > 0
                    if pos_i.sum() == 0:
                        continue

                    pos_props_i = props_i[pos_i]
                    pos_labels_i = labels_i[pos_i]
                    pos_gt_masks_i = gt_masks_i[pos_i] if gt_masks_i is not None else None

                    mask_features_i = self.mask_roi_pool(fpn4, [pos_props_i], image_size)
                    mask_logits_i = self.mask_head(mask_features_i)
                    mask_losses.append(
                        self._mask_loss(mask_logits_i, pos_props_i, pos_gt_masks_i, pos_labels_i)
                    )

                if mask_losses:
                    roi_losses["mask_loss"] = torch.stack(mask_losses).mean()
                else:
                    roi_losses["mask_loss"] = cls_scores.sum() * 0
            else:
                roi_losses["mask_loss"] = cls_scores.sum() * 0

            return [], roi_losses

        # ---- Inference ----
        detections = []
        offset = 0
        for i, props in enumerate(proposals):
            n = props.shape[0]
            c = cls_scores[offset:offset+n]
            b = bbox_pred[offset:offset+n]
            offset += n

            boxes, scores, det_labels = self._post_process(c, b, props, image_size)

            # Masks for top detections
            masks = None
            if boxes.shape[0] > 0:
                mask_features = self.mask_roi_pool(fpn4, [boxes], image_size)
                mask_logits   = self.mask_head(mask_features)           # (D, K, 28, 28)
                idx           = torch.arange(boxes.shape[0], device=boxes.device)
                masks         = (mask_logits[idx, det_labels] > self.mask_thresh)  # (D, 28, 28)

            detections.append({
                "boxes" : boxes,
                "labels": det_labels,
                "scores": scores,
                "masks" : masks,
            })

        return detections, {}