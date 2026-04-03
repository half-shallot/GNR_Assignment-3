"""
mask_rcnn.py  –  Full Mask R-CNN assembly + dataset + training script
Paper: He et al., ICCV 2017  (arXiv 1703.06870)

Architecture
------------
ResNet-101-FPN  backbone
↓
RPN  (anchors: 5 scales × 3 aspect ratios)
↓  proposals
Multi-Scale RoIAlign  (7×7 for box head, 14×14 for mask head)
↓
Box Head    →  L_cls  +  L_box
Mask Head   →  L_mask  (binary cross-entropy, decoupled from classification)

Usage
-----
    # Quick smoke test (random inputs, no real data needed)
    python mask_rcnn.py --mode test

    # Train on COCO-style dataset
    python mask_rcnn.py --mode train --data_dir /path/to/coco \
                        --epochs 8 --batch_size 2
"""

import os
import sys
import math
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

# Local modules
from back_bone import ResNet101FPN
from rpn      import RPN, RPNHead, AnchorGenerator
from roi_align import MultiScaleRoIAlign
from roi_head  import RoIHead, BoxHead, MaskHead

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


# ===========================================================================
# Model Assembly
# ===========================================================================

def build_mask_rcnn(num_classes=81, pretrained_backbone=True):
    """
    Assemble a complete Mask R-CNN with ResNet-101-FPN backbone.

    num_classes : including background (COCO = 80 + 1 = 81)
    """
    # ---- Backbone ----
    backbone = ResNet101FPN(pretrained=pretrained_backbone)
    out_ch   = backbone.out_channels   # 256

    # ---- Anchor Generator (5 levels, 3 aspect ratios each) ----
    # Paper §3.1: "RPN anchors span 5 scales and 3 aspect ratios"
    anchor_gen = AnchorGenerator(
        sizes        = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios= ((0.5, 1.0, 2.0),) * 5,
    )
    num_anchors = len(anchor_gen.aspect_ratios[0]) * len(anchor_gen.sizes[0])  # 3

    # ---- RPN ----
    rpn_head = RPNHead(out_ch, num_anchors)
    rpn = RPN(
        anchor_generator=anchor_gen,
        head=rpn_head,
        pre_nms_top_n_train=2000, pre_nms_top_n_test=1000,
        post_nms_top_n_train=2000, post_nms_top_n_test=1000,
        nms_thresh=0.7,
        fg_iou_thresh=0.7,  bg_iou_thresh=0.3,
        batch_size_per_image=256, positive_fraction=0.5,
    )

    # ---- RoI Pooling layers ----
    # Box head uses 7×7, mask head uses 14×14  (Figure 4)
    box_roi_pool  = MultiScaleRoIAlign(output_size=7,  sampling_ratio=2)
    mask_roi_pool = MultiScaleRoIAlign(output_size=14, sampling_ratio=2)

    # ---- Heads ----
    box_head  = BoxHead(in_channels=out_ch, roi_size=7,  num_classes=num_classes)
    mask_head = MaskHead(in_channels=out_ch,              num_classes=num_classes)

    # ---- RoI Head ----
    roi_head = RoIHead(
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        box_head=box_head,
        mask_head=mask_head,
        num_classes=num_classes,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        mask_thresh=0.5,
    )

    return MaskRCNN(backbone, rpn, roi_head)


# ===========================================================================
# Top-level Module
# ===========================================================================

class MaskRCNN(nn.Module):
    """
    End-to-end Mask R-CNN.

    Training forward pass returns a dict of losses.
    Inference forward pass returns a list of detection dicts.
    """

    def __init__(self, backbone, rpn, roi_head):
        super().__init__()
        self.backbone = backbone
        self.rpn      = rpn
        self.roi_head = roi_head

    def forward(self, images, targets=None):
        """
        images  : Tensor (N, 3, H, W)  – normalised
        targets : list[dict] during training, None during inference
          Each dict:
            'boxes'  : Tensor (G, 4)  xyxy image coords
            'labels' : Tensor (G,)    int64, 1-indexed class ids
            'masks'  : Tensor (G, H, W) bool  (optional)

        Returns
        -------
        Training  : dict of scalar loss tensors
        Inference : list of detection dicts (one per image)
        """
        if self.training and targets is None:
            raise ValueError("targets required during training")

        N, _, H, W = images.shape
        image_size  = (H, W)

        # ---- Extract FPN features ----
        features = self.backbone(images)     # [P2, P3, P4, P5, P6]

        # ---- RPN ----
        proposals, rpn_losses = self.rpn(
            features, image_size, targets
        )

        # Add GT boxes to proposals at training time
        if self.training and targets is not None:
            for i, tgt in enumerate(targets):
                proposals[i] = torch.cat([proposals[i], tgt["boxes"]], dim=0)

        # ---- RoI Head ----
        detections, roi_losses = self.roi_head(
            features, proposals, image_size, targets
        )

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        return detections


# ===========================================================================
# Minimal COCO-style Dataset
# ===========================================================================

class CocoInstanceDataset(Dataset):
    """
    Loads images + COCO-format JSON annotations.

    Expected annotation format (COCO):
    {
      "images": [{"id": int, "file_name": str, "height": int, "width": int}],
      "annotations": [{
          "id": int, "image_id": int, "category_id": int,
          "bbox": [x, y, w, h],
          "segmentation": [[x1,y1,...]]  (polygon)
      }],
      "categories": [{"id": int, "name": str}]
    }
    """

    def __init__(self, img_dir, ann_file, max_size=800, min_size=800):
        self.img_dir  = Path(img_dir)
        self.max_size = max_size
        self.min_size = min_size

        with open(ann_file) as f:
            coco = json.load(f)

        self.imgs = {img["id"]: img for img in coco["images"]}
        self.img_ids = list(self.imgs.keys())

        # Group annotations by image
        self.anns = {iid: [] for iid in self.img_ids}
        for ann in coco["annotations"]:
            if ann["image_id"] in self.anns:
                self.anns[ann["image_id"]].append(ann)

        # Build category id → contiguous index (1-based)
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        self.cat2idx = {c["id"]: i + 1 for i, c in enumerate(cats)}

        # ImageNet mean/std
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std  = torch.tensor([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img_ids)

    def _resize(self, img):
        """Resize so shorter side = min_size, longer side ≤ max_size."""
        h, w = img.shape[-2], img.shape[-1]
        scale = self.min_size / min(h, w)
        if scale * max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        return TF.resize(img, [nh, nw]), scale

    def _poly_to_mask(self, poly, H, W):
        """Rasterise a COCO polygon to a boolean mask."""
        from PIL import Image, ImageDraw
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for seg in poly:
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
        return torch.from_numpy(np.array(mask, dtype=bool))

    def _segmentation_to_mask(self, segmentation, H, W, scale):
        """Convert COCO segmentation (polygon or RLE) into a resized bool mask."""
        if not segmentation:
            return torch.zeros(H, W, dtype=torch.bool)

        # Polygon format: list[list[float]]
        if isinstance(segmentation, list):
            scaled_poly = []
            for seg in segmentation:
                if not isinstance(seg, (list, tuple)):
                    continue
                try:
                    scaled_seg = [float(x) * scale for x in seg]
                except (TypeError, ValueError):
                    continue
                if len(scaled_seg) >= 6:
                    scaled_poly.append(scaled_seg)
            if not scaled_poly:
                return torch.zeros(H, W, dtype=torch.bool)
            return self._poly_to_mask(scaled_poly, H, W)

        # RLE format: {"counts": ..., "size": [h, w]}
        if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
            try:
                from pycocotools import mask as mask_utils
            except ImportError:
                # If pycocotools is unavailable, gracefully skip RLE masks.
                return torch.zeros(H, W, dtype=torch.bool)

            try:
                rle = segmentation
                mask_np = mask_utils.decode(rle)
                if mask_np.ndim == 3:
                    mask_np = mask_np[..., 0]
                mask_t = torch.from_numpy(mask_np.astype(np.uint8))
                mask_t = mask_t.unsqueeze(0).unsqueeze(0).float()
                mask_t = torch.nn.functional.interpolate(mask_t, size=(H, W), mode="nearest")
                return mask_t[0, 0] > 0.5
            except Exception:
                return torch.zeros(H, W, dtype=torch.bool)

        return torch.zeros(H, W, dtype=torch.bool)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.imgs[img_id]
        H0, W0   = img_info["height"], img_info["width"]

        # Load image
        img_path = self.img_dir / img_info["file_name"]
        pil_img  = Image.open(img_path).convert("RGB")
        img_t    = TF.to_tensor(pil_img)                         # (3, H, W)
        img_t, scale = self._resize(img_t)
        img_t    = TF.normalize(img_t, self.mean, self.std)
        H1, W1   = img_t.shape[-2], img_t.shape[-1]

        anns = self.anns[img_id]
        boxes, labels, masks = [], [], []

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            # Scale to resized image
            x1, x2 = x1 * scale, x2 * scale
            y1, y2 = y1 * scale, y2 * scale
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2idx[ann["category_id"]])

            masks.append(self._segmentation_to_mask(ann.get("segmentation"), H1, W1, scale))

        target = {
            "boxes" : torch.tensor(boxes,  dtype=torch.float32) if boxes
                      else torch.zeros(0, 4, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)    if labels
                      else torch.zeros(0, dtype=torch.long),
            "masks" : torch.stack(masks) if masks
                      else torch.zeros(0, H1, W1, dtype=torch.bool),
            "image_id": torch.tensor([img_id]),
        }
        return img_t, target


def collate_fn(batch):
    images  = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images = []
    padded_targets = []

    for img, tgt in zip(images, targets):
        h, w = img.shape[-2], img.shape[-1]
        pad = (0, max_w - w, 0, max_h - h)  # left, right, top, bottom

        padded_images.append(F.pad(img, pad, mode="constant", value=0.0))

        tgt = dict(tgt)
        if "masks" in tgt and torch.is_tensor(tgt["masks"]) and tgt["masks"].numel() > 0:
            tgt["masks"] = F.pad(
                tgt["masks"].float(), pad, mode="constant", value=0.0
            ) > 0.5

        padded_targets.append(tgt)

    return torch.stack(padded_images), padded_targets


# ===========================================================================
# Training Loop
# ===========================================================================

def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=50):
    model.train()
    total_loss = 0.
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (commonly used with Mask R-CNN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % print_freq == 0:
            loss_str = "  ".join(f"{k}: {v.item():.4f}"
                                 for k, v in loss_dict.items())
            print(f"Epoch [{epoch}] Step [{i+1}/{len(loader)}]  {loss_str}"
                  f"  total: {loss.item():.4f}")

    return total_loss / len(loader)


def build_optimizer(model, lr=0.02, momentum=0.9, weight_decay=1e-4):
    """
    §3.1: lr=0.02, weight_decay=0.0001, momentum=0.9
    Backbone uses 0.1× lr (standard practice for fine-tuning).
    """
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    other_params    = [p for p in model.parameters()
                       if id(p) not in backbone_ids]
    return optim.SGD(
        [{"params": backbone_params, "lr": lr * 0.1},
         {"params": other_params,    "lr": lr}],
        momentum=momentum, weight_decay=weight_decay
    )


def build_scheduler(optimizer, milestones, gamma=0.1):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)


# ===========================================================================
# Smoke Test (random tensors, no data required)
# ===========================================================================

def smoke_test():
    print("=" * 60)
    print("Mask R-CNN  –  smoke test (random inputs)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_mask_rcnn(num_classes=81, pretrained_backbone=False)
    model = model.to(device)

    # ---- Training forward pass ----
    model.train()
    B, C, H, W = 2, 3, 800, 800
    images = torch.rand(B, C, H, W, device=device)
    targets = []
    for _ in range(B):
        n_gt = 3
        x1y1 = torch.rand(n_gt, 2, device=device) * 400
        x2y2 = x1y1 + torch.rand(n_gt, 2, device=device) * 200 + 10
        boxes  = torch.cat([x1y1, x2y2], dim=1).clamp(0, W-1)
        labels = torch.randint(1, 81, (n_gt,), device=device)
        masks  = torch.rand(n_gt, H, W, device=device) > 0.5
        targets.append({"boxes": boxes, "labels": labels, "masks": masks})

    loss_dict = model(images, targets)
    total_loss = sum(loss_dict.values())
    print("\n[Train] Losses:")
    for k, v in loss_dict.items():
        print(f"  {k:25s}: {v.item():.4f}")
    print(f"  {'total':25s}: {total_loss.item():.4f}")

    # ---- Backward ----
    total_loss.backward()
    print("\n[Train] Backward pass OK")

    # ---- Inference forward pass ----
    model.eval()
    with torch.no_grad():
        detections = model(images)

    print("\n[Inference] Detections per image:")
    for i, det in enumerate(detections):
        n = det["boxes"].shape[0]
        print(f"  Image {i}: {n} detections")
        if n > 0:
            print(f"    boxes  shape : {det['boxes'].shape}")
            print(f"    labels shape : {det['labels'].shape}")
            print(f"    scores shape : {det['scores'].shape}")
            if det["masks"] is not None:
                print(f"    masks  shape : {det['masks'].shape}")

    # ---- Parameter count ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters : {total_params/1e6:.1f} M")
    print(f"Trainable params : {trainable/1e6:.1f} M")
    print("\nSmoke test PASSED ✓")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser("Mask R-CNN  (PyTorch, ResNet-101-FPN)")
    parser.add_argument("--mode",       default="test",
                        choices=["test", "train"])
    parser.add_argument("--data_dir",   default="./coco/images/train2017")
    parser.add_argument("--ann_file",   default="./coco/annotations/instances_train2017.json")
    parser.add_argument("--epochs",     type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr",         type=float, default=0.02)
    parser.add_argument("--num_classes",type=int, default=81)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--resume",     default=None)
    args = parser.parse_args()

    if args.mode == "test":
        smoke_test()
        return

    # ---- Training ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_mask_rcnn(num_classes=args.num_classes).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    dataset = CocoInstanceDataset(args.data_dir, args.ann_file)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=True
    )

    optimizer = build_optimizer(model, lr=args.lr)
    # Paper §3.1: lr drops by 10× at 120k and 160k iterations
    # Approximate as epoch milestones for the 12-epoch "1×" schedule
    scheduler = build_scheduler(optimizer, milestones=[8, 11])

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, loader, device, epoch)
        scheduler.step()
        print(f"\nEpoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[-1]['lr']:.6f}\n")

        ckpt_path = Path(args.output_dir) / f"mask_rcnn_epoch{epoch:02d}.pth"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()