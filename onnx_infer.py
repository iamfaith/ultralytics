#!/usr/bin/env python3
"""
Standalone ONNX inference script (no ultralytics dependency).
Usage example:
  python onnx_infer.py --model yolov5nu.onnx --source /home/faith/fux.png --device cpu --conf 0.5 --out fux_with_boxes.png

Notes:
- Requires `onnxruntime`, `opencv-python`, `numpy`.
- For GPU use `--device cuda` and install `onnxruntime-gpu`.
"""
import argparse
import sys
import os
import time
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception as e:
    print("onnxruntime not installed. Install via: pip install onnxruntime (or onnxruntime-gpu)")
    raise


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while keeping aspect ratio. Returns (img, ratio, pad)
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def xywh2xyxy(x):
    # x: [x_center, y_center, w, h]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    # boxes: (N,4) x1,y1,x2,y2
    # scores: (N,)
    if len(boxes) == 0:
        return []
    boxes_xywh = []
    for x1, y1, x2, y2 in boxes:
        boxes_xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    indices = indices.flatten().tolist()
    return indices


def load_labels(labels_path):
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            names = [x.strip() for x in f.read().strip().splitlines() if x.strip()]
        return names
    # fallback to COCO names (80 classes)
    return [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
        'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
        'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
        'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
        'sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]



def origin_postprocess(outputs, conf_thres, iou_thres, names, img, img0):

    import torch
    from ultralytics.utils import nms as ul_nms
    from ultralytics.utils import ops as ul_ops
    # Convert ONNX output to torch tensor in the format expected by repo NMS (batch, C, N)
    pred_tensor = torch.tensor(outputs[0]) if not isinstance(outputs[0], torch.Tensor) else outputs[0]

    # Call Ultralytics NMS (returns list of tensors, one per image)
    preds = ul_nms.non_max_suppression(
        pred_tensor,
        conf_thres,
        iou_thres,
        None,
        False,
        max_det=300,
        nc=0,
        end2end=False,
    )

    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    if pred is None or pred.numel() == 0:
        print(f"No detections with confidence > {conf_thres}")
        return

    # Scale boxes from model input shape to original image
    pred[:, :4] = ul_ops.scale_boxes(img.shape[2:], pred[:, :4], img0.shape)

    # Draw boxes
    for det in pred.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls = int(cls)
        label = f"{names[cls] if cls < len(names) else cls} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img0, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    out = "original.png"
    cv2.imwrite(out, img0)
    print(f"Saved {out}")


def run(model_path, source, device='cpu', img_size=640, conf_thres=0.5, iou_thres=0.45, labels=None, out='out.png'):
    providers = ['CPUExecutionProvider']
    if device.lower() in ('cuda', 'gpu'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # If input shape contains dynamic dims, fallback to img_size
    try:
        _, c, h, w = [int(x) if isinstance(x, (int, np.integer)) else None for x in input_shape]
        if h is None or w is None:
            h = w = img_size
    except Exception:
        h = w = img_size

    names = load_labels(labels)

    img_bgr = cv2.imread(source)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {source}")

    img0 = img_bgr.copy()
    img, ratio, (pad_w, pad_h) = letterbox(img_bgr, new_shape=(h, w))
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)  # add batch

    # Run inference
    t0 = time.time()
    outputs = sess.run(None, {input_name: img})
    t1 = time.time()
    print(f"Inference time: {(t1 - t0) * 1000:.1f} ms")


    origin_postprocess(outputs, conf_thres, iou_thres, names, img, img0.copy())

    pred = outputs[0]
    if pred.ndim == 3 and pred.shape[1] > pred.shape[2]:
        pred = np.transpose(pred, (0, 2, 1))
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    nc = pred.shape[1] - 5
    if nc < 1:
        raise RuntimeError("Unexpected model output shape, cannot determine number of classes")
    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_confs = cls_scores.max(axis=1)
    # Candidate if either objectness or class confidence exceeds threshold (matches repo's amax behavior)
    cand = np.maximum(obj_conf, cls_confs)
    mask = cand > conf_thres
    boxes = boxes[mask]
    # Use class confidence for scoring/labels (matches Ultralytics postprocess)
    scores = cls_confs[mask]
    cls_ids = cls_ids[mask]
    if boxes.shape[0] == 0:
        print(f"No detections with confidence > {conf_thres}")
        cv2.imwrite(out, img0)
        print(f"Saved {out}")
        return
    boxes_xyxy = xywh2xyxy(boxes.copy())
    # remove padding and scale back to original image
    boxes_xyxy[:, [0, 2]] -= pad_w
    boxes_xyxy[:, [1, 3]] -= pad_h
    boxes_xyxy /= ratio
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, img0.shape[1])
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, img0.shape[0])
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, img0.shape[1])
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, img0.shape[0])

    # Class-aware NMS to mimic Ultralytics behavior: run NMS per-class and merge
    keep_global = []
    unique_classes = np.unique(cls_ids)
    for c in unique_classes:
        idxs = np.where(cls_ids == c)[0]
        if idxs.size == 0:
            continue
        cls_boxes = boxes_xyxy[idxs]
        cls_scores = scores[idxs]
        # cv2.dnn.NMSBoxes expects boxes in [x,y,w,h]
        cls_boxes_xywh = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in cls_boxes]
        if len(cls_boxes_xywh) == 0:
            continue
        keep = cv2.dnn.NMSBoxes(cls_boxes_xywh, cls_scores.tolist(), score_threshold=0.0, nms_threshold=iou_thres)
        if len(keep) == 0:
            continue
        keep = keep.flatten().tolist()
        # map back to global indices
        keep_global.extend(idxs[k] for k in keep)

    if len(keep_global) == 0:
        print("All detections removed by NMS")
        cv2.imwrite(out, img0)
        print(f"Saved {out}")
        return

    # optional: sort by score desc and keep top max_det (mimic repo)
    keep_global = list(set(keep_global))
    keep_global.sort(key=lambda i: float(scores[i]), reverse=True)
    max_det = 300
    keep_global = keep_global[:max_det]

    for i in keep_global:
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cls = int(cls_ids[i])
        score = float(scores[i])
        color = (0, 255, 0)
        label = f"{names[cls] if cls < len(names) else cls} {score:.2f}"
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img0, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out, img0)
    print(f"Saved numpy {out}")
    


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', required=False, help='ONNX model path', default='yolov5nu.onnx')
    p.add_argument('--source', '-s', required=False, help='Image path', default='/home/faith/fux.png')
    p.add_argument('--device', '-d', default='cpu', help='cpu or cuda')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--labels', help='optional labels txt file')
    p.add_argument('--out', default='out.png', help='output image path')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.model, args.source, device=args.device, img_size=args.img_size, conf_thres=args.conf, iou_thres=args.iou, labels=args.labels, out=args.out)
