from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

base = Path('.')
model_path = base / 'yolov8n_seeds.pt'
print('model exists', model_path.exists())
img_path = None
for root in ['dataset/train/images', 'dataset_yolo/images/train', 'dataset/train', '.']:
    p = base / root
    if p.exists():
        for ext in ['jpg', 'jpeg', 'png']:
            for f in p.rglob(f'*.{ext}'):
                img_path = f
                break
            if img_path:
                break
    if img_path:
        break
print('img_path', img_path)
if not img_path:
    raise SystemExit('no sample image found')
img = Image.open(img_path).convert('RGB')
print('img size', img.size)
model = YOLO(str(model_path))
res = model.predict(np.asarray(img), verbose=False, conf=0.2, imgsz=640, iou=0.45, max_det=32, augment=False)
print('results len', len(res))
if res:
    r = res[0]
    if r.boxes is not None:
        print('xyxy', r.boxes.xyxy.cpu().numpy())
        print('cls', r.boxes.cls.cpu().numpy())
        print('conf', r.boxes.conf.cpu().numpy())
    else:
        print('no boxes')
