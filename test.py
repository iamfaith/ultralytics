from ultralytics import YOLO

# Load a COCO-pretrained YOLOv5n model
model = YOLO("yolov5nu.onnx")

# Display model information (optional)
# model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv5n model on the 'bus.jpg' image
# results = model("/home/faith/fux.png")

results = model.predict(source="/home/faith/fux.png", device="cpu")  # 用GPU就写 device=0

# 绘制检测框并保存结果图像（以 BGR 输出，并按置信度过滤）
res = results[0]
import cv2
import numpy as np

image_path = "/home/faith/fux.png"
# 以 BGR 读取原图（OpenCV 默认 BGR）
img_bgr = cv2.imread(image_path)
if img_bgr is None:
	# 回退：如果未能从磁盘读取，则尝试使用推理时的原始图像（通常为 RGB）并转换为 BGR
	try:
		orig = res.orig_img
		orig_np = np.array(orig)
		img_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
	except Exception:
		raise RuntimeError(f"无法读取图像: {image_path}")

# 从结果中获取边框、置信度和类别（可能为 torch tensor 或 numpy）
def to_numpy(x):
	if x is None:
		return None
	try:
		return x.cpu().numpy()
	except Exception:
		return np.array(x)

boxes = to_numpy(getattr(res.boxes, 'xyxy', None))
confs = to_numpy(getattr(res.boxes, 'conf', None))
clss = to_numpy(getattr(res.boxes, 'cls', None))
names = getattr(res, 'names', None) or getattr(model, 'names', {})

thres = 0.5
if boxes is not None and confs is not None:
	confs_flat = confs.flatten()
	mask = confs_flat > thres
	if mask.sum() == 0:
		print(f"No detections with confidence > {thres}")
	else:
		for i, keep in enumerate(mask):
			if not keep:
				continue
			box = boxes[i]
			conf = confs_flat[i]
			cls = int(clss[i]) if clss is not None else None
			x1, y1, x2, y2 = map(int, box)
			color = (0, 255, 0)
			cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
			label = f"{names[cls] if (cls is not None and names and cls in names) else cls} {conf:.2f}"
			(w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
			cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

out_path = "fux_with_boxes.png"
cv2.imwrite(out_path, img_bgr)
print(f"Saved {out_path} (BGR)")



# model.export(format="onnx")