"""
test_mask_rcnn.py  –  Inference with Detectron2 Visualizer
Requires:
    pip install torch torchvision opencv-python Pillow
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

Usage
-----
    python test_mask_rcnn.py --weights model.pth --input image.jpg
    python test_mask_rcnn.py --weights model.pth --input ./images/ --output ./results/
    python test_mask_rcnn.py --weights model.pth --input img.jpg --score_thresh 0.5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from mask_rcnn import build_mask_rcnn

# Detectron2 – visualization only
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BitMasks, Instances
from detectron2.utils.visualizer import ColorMode, Visualizer

# ---------------------------------------------------------------------------
# COCO metadata  (80 thing classes, 0-indexed for detectron2)
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

_META_NAME = "coco_inference"
if _META_NAME not in MetadataCatalog:
    MetadataCatalog.get(_META_NAME).set(thing_classes=COCO_CLASSES)
_META = MetadataCatalog.get(_META_NAME)

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def preprocess(pil_img, max_size=1333, min_size=800):
    img = TF.to_tensor(pil_img.convert("RGB"))
    img = TF.normalize(img, _MEAN, _STD)
    _, h, w = img.shape
    scale = min_size / min(h, w)
    if scale * max(h, w) > max_size:
        scale = max_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = TF.resize(img, [nh, nw])
    return img.unsqueeze(0), scale


def paste_box_masks_to_image(box_masks, boxes, image_hw, mask_thresh=0.5):
    """Paste box-relative 28x28 masks into full-image binary masks."""
    h, w = image_hw
    n = int(box_masks.shape[0])
    out = np.zeros((n, h, w), dtype=bool)

    if n == 0:
        return out

    masks_np = box_masks.detach().cpu().numpy()
    boxes_np = boxes.detach().cpu().numpy()

    for i in range(n):
        x1, y1, x2, y2 = boxes_np[i]
        x1 = int(np.floor(max(0.0, x1)))
        y1 = int(np.floor(max(0.0, y1)))
        x2 = int(np.ceil(min(float(w), x2)))
        y2 = int(np.ceil(min(float(h), y2)))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        resized = cv2.resize(
            masks_np[i].astype(np.float32),
            (bw, bh),
            interpolation=cv2.INTER_LINEAR,
        )
        out[i, y1:y2, x1:x2] = resized > mask_thresh

    return out


def expand_boxes_xyxy(boxes, image_hw, expand_pct=0.0, expand_px=0.0):
    """Expand xyxy boxes around center, then clip to image bounds."""
    if boxes.numel() == 0 or (expand_pct <= 0 and expand_px <= 0):
        return boxes

    h, w = image_hw
    expanded = boxes.clone().float()

    widths = expanded[:, 2] - expanded[:, 0]
    heights = expanded[:, 3] - expanded[:, 1]
    cx = (expanded[:, 0] + expanded[:, 2]) * 0.5
    cy = (expanded[:, 1] + expanded[:, 3]) * 0.5

    half_w = widths * 0.5
    half_h = heights * 0.5
    half_w = half_w * (1.0 + float(expand_pct)) + float(expand_px)
    half_h = half_h * (1.0 + float(expand_pct)) + float(expand_px)

    expanded[:, 0] = (cx - half_w).clamp(min=0.0, max=float(w))
    expanded[:, 1] = (cy - half_h).clamp(min=0.0, max=float(h))
    expanded[:, 2] = (cx + half_w).clamp(min=0.0, max=float(w))
    expanded[:, 3] = (cy + half_h).clamp(min=0.0, max=float(h))
    return expanded

# ---------------------------------------------------------------------------
# Inference + Detectron2 visualisation
# ---------------------------------------------------------------------------
def run_inference(
    model,
    img_path,
    device,
    score_thresh,
    output_dir,
    use_coco_names,
    masks_only=True,
    box_expand_pct=0.0,
    box_expand_px=0.0,
    full_image_box=False,
):
    pil     = Image.open(img_path).convert("RGB")
    orig_np = np.array(pil)                        # H×W×3 uint8 RGB
    orig_h, orig_w = orig_np.shape[:2]

    tensor, scale = preprocess(pil)
    with torch.no_grad():
        detections = model(tensor.to(device))

    det  = detections[0]
    keep = det["scores"] >= score_thresh

    boxes_s = det["boxes"][keep].cpu()
    labels  = det["labels"][keep].cpu()
    scores  = det["scores"][keep].cpu()
    n       = int(keep.sum())
    print(f"  {img_path.name}: {n} detection(s)  (thresh={score_thresh})")

    # Scale boxes back to original image space
    boxes_orig = boxes_s / scale
    boxes_vis = expand_boxes_xyxy(
        boxes_orig,
        (orig_h, orig_w),
        expand_pct=box_expand_pct,
        expand_px=box_expand_px,
    )

    if full_image_box and boxes_vis.numel() > 0:
        boxes_vis[:, 0] = 0.0
        boxes_vis[:, 1] = 0.0
        boxes_vis[:, 2] = float(orig_w)
        boxes_vis[:, 3] = float(orig_h)

    # Paste per-box masks into original image space
    masks_orig = None
    raw_masks  = det.get("masks")
    if raw_masks is not None and n > 0:
        masks_box = raw_masks[keep]
        if masks_box.ndim == 4 and masks_box.shape[1] == 1:
            masks_box = masks_box[:, 0]
        masks_orig = paste_box_masks_to_image(
            masks_box,
            boxes_vis,
            (orig_h, orig_w),
            mask_thresh=0.5,
        )

    # Build Detectron2 Instances
    instances = Instances((orig_h, orig_w))
    if not masks_only:
        instances.pred_boxes = Boxes(boxes_vis)
    instances.scores       = scores
    # torchvision uses 1-indexed labels (0 = background); detectron2 is 0-indexed
    instances.pred_classes = (labels - 1).clamp(min=0)
    if masks_orig is not None:
        instances.pred_masks = BitMasks(torch.from_numpy(masks_orig))

    # Detectron2 Visualizer – does filled masks + contours + labels
    meta = _META if use_coco_names else MetadataCatalog.get("__empty__")
    vis  = Visualizer(
        orig_np,
        metadata      = meta,
        scale         = 1.0,
        instance_mode = ColorMode.SEGMENTATION,   # unique colour per instance
    )
    result = vis.draw_instance_predictions(instances.to("cpu")).get_image()

    out_path = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
    cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"  Saved → {out_path}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Mask R-CNN – Detectron2 visualisation")
    parser.add_argument("--weights",       required=True)
    parser.add_argument("--input",         required=True,
                        help="Image file or directory")
    parser.add_argument("--output",        default="./results")
    parser.add_argument("--num_classes",   type=int, default=81,
                        help="Including background. COCO = 81 (default)")
    parser.add_argument("--score_thresh",  type=float, default=0.5)
    parser.add_argument("--no_coco_names", action="store_true",
                        help="Show numeric IDs instead of COCO class names")
    parser.add_argument(
        "--show_boxes",
        action="store_true",
        help="Draw bounding boxes in addition to masks",
    )
    parser.add_argument(
        "--box_expand_pct",
        type=float,
        default=0.0,
        help="Expand each box by this fraction per side aggregate (e.g. 0.1 = 10%)",
    )
    parser.add_argument(
        "--box_expand_px",
        type=float,
        default=0.0,
        help="Expand each box by this many pixels on each side",
    )
    parser.add_argument(
        "--full_image_box",
        action="store_true",
        help="Use full image as the projection box for every mask",
    )
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)

    model = build_mask_rcnn(num_classes=args.num_classes, pretrained_backbone=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"Device : {device}  |  weights : {args.weights}")
    print(f"classes: {args.num_classes}  |  thresh : {args.score_thresh}\n")
    print(f"visual: {'masks+boxes' if args.show_boxes else 'masks-only'}\n")
    print(f"box expand: pct={args.box_expand_pct}  px={args.box_expand_px}\n")
    print(f"full-image box for masks: {args.full_image_box}\n")

    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    p       = Path(args.input)
    imgs    = sorted(p.iterdir() if p.is_dir() else [p], key=lambda x: x.name)
    imgs    = [i for i in imgs if i.suffix.lower() in exts]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running on {len(imgs)} image(s)...\n")
    for img_path in imgs:
        run_inference(model, img_path, device, args.score_thresh,
                      out_dir, not args.no_coco_names,
                      masks_only=not args.show_boxes,
                      box_expand_pct=args.box_expand_pct,
                      box_expand_px=args.box_expand_px,
                      full_image_box=args.full_image_box)

    print(f"Done. Results → {out_dir}/")

if __name__ == "__main__":
    main()