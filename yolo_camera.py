from pathlib import Path
from collections import deque, Counter
import threading
import time

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
TRAINED_YOLO = BASE_DIR / "yolov8n_seeds.pt"
YOLO_CONF_THRESHOLD = 0.22
CLASS_CONF_THRESHOLD = 0.62
MIN_BOX_AREA_RATIO = 0.0005
MAX_BOX_AREA_RATIO = 0.98
MIN_REFINED_BOX_AREA_RATIO = 0.0003
MAX_REFINED_BOX_AREA_RATIO = 0.98
STABLE_FRAMES = 5
STABLE_MIN_VOTES = 3
INFERENCE_WIDTH = 640
INFERENCE_INTERVAL_SECONDS = 0.12
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
IMG_SIZE = (224, 224)

# When debugging, show raw YOLO detections (before CNN filtering/refinement)
DEBUG_SHOW_YOLO = True
DEBUG_VERBOSE = True
DEBUG_SAVE_FAILED_CROPS = True
DEBUG_DIR = BASE_DIR / "debug_crops"
DEBUG_DIR.mkdir(exist_ok=True)
_debug_save_counter = 0

LABEL_ALIASES = {
    "Bitter melon": "Bitter_melon",
    "Coriander leaves": "Coriander_leaves",
    "cucumber_green _long": "cucumber_green_long",
}


def norm_label(label: str) -> str:
    return "_".join(label.strip().lower().replace("_", " ").split())


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def _seed_key(label: str) -> str:
    return "_".join(str(label or "").lower().replace("_", " ").split())


def build_seed_vocabulary(classes: list[str]) -> set[str]:
    toks: set[str] = set()
    for c in classes:
        c_low = c.lower().strip()
        toks.add(c_low)
        toks.add(c_low.replace("_", " "))
        toks.add(c_low.replace(" ", "_"))
        for piece in c_low.replace("_", " ").split():
            if len(piece) > 2:
                toks.add(piece)
    toks.update({"seed", "seeds", "grain", "grains", "kernel", "bean"})
    return toks


def is_seed_label(label: str, known_labels: set[str], token_vocab: set[str]) -> bool:
    normalized = norm_label(label)
    return normalized == "seed" or normalized in known_labels or normalized in token_vocab


def load_classes() -> list[str]:
    classes_path = BASE_DIR / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class list: {classes_path}")
    with classes_path.open("r", encoding="utf-8") as f:
        return [canonical_label(line.strip()) for line in f if line.strip()]


def load_cnn_models() -> list[tf.keras.Model]:
    models = []
    for name in ("model1.h5", "model2.h5", "model3.h5"):
        path = BASE_DIR / name
        if path.exists():
            try:
                models.append(tf.keras.models.load_model(str(path), compile=False))
                print(f"Loaded CNN model: {name}")
            except Exception as exc:
                print(f"Skipping unreadable CNN model {name}: {exc}")
    if not models:
        print("Warning: No CNN models loaded. Running YOLO-only inference.")
        return models


def classify_seed(crop: np.ndarray, cnn_models: list[tf.keras.Model], classes: list[str]) -> tuple[str, float]:
    if crop.size == 0:
        return "seed", 0.0

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = rgb.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    preds = []
    for cnn in cnn_models:
        pred = cnn.predict(x, verbose=0)[0]
        preds.append(np.asarray(pred, dtype="float32"))
    n_classes = max(len(pred) for pred in preds)
    aligned = []
    for pred in preds:
        if len(pred) < n_classes:
            pred = np.pad(pred, (0, n_classes - len(pred)))
        aligned.append(pred[:n_classes])

    avg = np.mean(np.stack(aligned, axis=0), axis=0)
    if avg.min() < 0 or avg.sum() > 1.5:
        exp = np.exp(avg - avg.max())
        avg = exp / exp.sum()

    cls_id = int(np.argmax(avg))
    label = canonical_label(classes[cls_id]) if cls_id < len(classes) else "seed"
    return label, float(avg[cls_id])


def draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(24, y1 - 10)
    cv2.putText(
        frame,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_status(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )


def draw_debug_overlay(frame: np.ndarray, lines: list[str]) -> None:
    """Draw a small debug box with detection info and confidences."""
    x, y = 12, 60
    line_height = 22
    margin = 8
    for i, line in enumerate(lines):
        y_pos = y + i * line_height
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        width = text_size[0] + margin * 2
        height = text_size[1] + margin
        cv2.rectangle(frame, (x - margin, y_pos - margin), (x + width, y_pos + height), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            frame,
            line,
            (x, y_pos + text_size[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def choose_best_camera_detections(detections: list[dict]) -> list[dict]:
    if len(detections) <= 1:
        return detections

    kept: list[dict] = []
    for det in sorted(detections, key=lambda d: (float(d.get("score", 0.0)), float(d.get("class_conf", 0.0))), reverse=True):
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        if all(bbox_iou(bbox, old["bbox"]) < 0.5 for old in kept):
            kept.append(det)
    return kept


def resize_for_inference(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    h, w = frame.shape[:2]
    if w <= INFERENCE_WIDTH:
        return frame, 1.0, 1.0
    scale = INFERENCE_WIDTH / float(w)
    infer_h = max(1, int(h * scale))
    infer_frame = cv2.resize(frame, (INFERENCE_WIDTH, infer_h), interpolation=cv2.INTER_AREA)
    return infer_frame, w / float(INFERENCE_WIDTH), h / float(infer_h)


def refine_seed_box(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Tighten YOLO's broad seed box with several foreground proposals."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h < 20 or crop_w < 20:
        return None

    def clean_mask(raw_mask: np.ndarray) -> np.ndarray:
        mask_u8 = raw_mask.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask_u8

    def candidate_boxes(mask_u8: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        crop_area = float(crop_w * crop_h)
        min_area = max(60.0, crop_area * MIN_REFINED_BOX_AREA_RATIO)
        max_area = crop_area * MAX_REFINED_BOX_AREA_RATIO
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rw < 5 or rh < 5:
                continue
            rect_ratio = (rw * rh) / crop_area
            if rect_ratio < MIN_REFINED_BOX_AREA_RATIO or rect_ratio > MAX_REFINED_BOX_AREA_RATIO:
                continue
            fill = area / float(max(1, rw * rh))
            aspect = max(rw / float(rh), rh / float(rw))
            cx = (rx + rw / 2.0) / crop_w
            cy = (ry + rh / 2.0) / crop_h
            center_penalty = abs(cx - 0.5) + abs(cy - 0.5)
            touches = int(rx <= 1) + int(ry <= 1) + int(rx + rw >= crop_w - 2) + int(ry + rh >= crop_h - 2)
            score = area * max(fill, 0.15) / max(aspect, 1.0) / (1.0 + center_penalty + touches)
            boxes.append((rx, ry, rw, rh, score))
        return boxes

    border = max(4, min(crop_h, crop_w) // 18)
    border_pixels = np.concatenate(
        [
            crop[:border, :, :].reshape(-1, 3),
            crop[-border:, :, :].reshape(-1, 3),
            crop[:, :border, :].reshape(-1, 3),
            crop[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border_pixels, axis=0)

    masks = []
    diff = np.linalg.norm(crop.astype(np.float32) - bg, axis=2)
    masks.append(diff > max(28.0, float(np.std(border_pixels)) * 1.15))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.extend([otsu_dark > 0, otsu_light > 0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    border_hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)[:, 0, :]
    sat_bg = float(np.median(border_hsv[:, 1]))
    val_bg = float(np.median(border_hsv[:, 2]))
    sat_diff = np.abs(hsv[:, :, 1].astype(np.float32) - sat_bg)
    val_diff = np.abs(hsv[:, :, 2].astype(np.float32) - val_bg)
    masks.append((sat_diff > 28) | (val_diff > 28))

    if crop_w >= 60 and crop_h >= 60:
        grab_mask = np.zeros((crop_h, crop_w), np.uint8)
        margin_x = max(3, int(crop_w * 0.04))
        margin_y = max(3, int(crop_h * 0.04))
        rect = (margin_x, margin_y, crop_w - 2 * margin_x, crop_h - 2 * margin_y)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(crop, grab_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
            masks.append((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD))
        except Exception:
            pass

    boxes = []
    for raw in masks:
        area = float(np.mean(raw))
        if MIN_REFINED_BOX_AREA_RATIO <= area <= MAX_REFINED_BOX_AREA_RATIO:
            boxes.extend(candidate_boxes(clean_mask(raw)))

    if not boxes:
        return None

    rx, ry, rw, rh, _score = max(boxes, key=lambda item: item[4])
    pad = max(6, int(min(rw, rh) * 0.12))
    rx1 = max(0, rx - pad)
    ry1 = max(0, ry - pad)
    rx2 = min(crop_w - 1, rx + rw + pad)
    ry2 = min(crop_h - 1, ry + rh + pad)
    refined_ratio = ((rx2 - rx1) * (ry2 - ry1)) / float(crop_w * crop_h)
    if refined_ratio < MIN_REFINED_BOX_AREA_RATIO or refined_ratio > MAX_REFINED_BOX_AREA_RATIO:
        return None

    return x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2


if not TRAINED_YOLO.exists():
    raise FileNotFoundError(
        f"Seed-only YOLO weights not found: {TRAINED_YOLO}. "
        "Train YOLO on seeds first; the seed-specific yolov8n_seeds.pt weights are required."
    )

yolo_model = YOLO(str(TRAINED_YOLO))
class_names = load_classes()
cnn_models = load_cnn_models()

if hasattr(yolo_model, 'names') and yolo_model.names:
    model_name_keys = {_seed_key(str(v)) for v in yolo_model.names.values()}
    dataset_name_keys = {_seed_key(c) for c in class_names}
    if not model_name_keys.issubset(dataset_name_keys):
        raise RuntimeError(
            "Loaded YOLO model classes do not match classes.txt. "
            "Retrain YOLO with train_yolo_seeds.py or use the correct seed-only weights."
        )

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened. Check camera permission/device index.")

print(f"Loaded seed-only YOLO model: {TRAINED_YOLO}")
print(f"Loaded {len(cnn_models)} CNN classifier model(s).")
print(f"Loaded {len(class_names)} seed class names from classes.txt.")
print(f"YOLO seed threshold: {YOLO_CONF_THRESHOLD:.2f}")
print(f"Database seed class threshold: {CLASS_CONF_THRESHOLD:.2f}")
if hasattr(yolo_model, 'names') and yolo_model.names:
    yolo_names = [yolo_model.names.get(i, str(i)) for i in range(min(len(yolo_model.names), 12))]
    print(f"YOLO model names sample: {yolo_names}")
print("Press ESC to exit.")

known_seed_labels = {norm_label(c) for c in class_names}
seed_tokens = build_seed_vocabulary(class_names)


def analyze_frame(frame: np.ndarray) -> dict:
    infer_frame, sx, sy = resize_for_inference(frame)
    results = yolo_model.predict(
        infer_frame,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
        imgsz=INFERENCE_WIDTH,
        iou=0.45,
        max_det=32,
        augment=False,
    )

    detections = []
    yolo_raw: list[dict] = []
    yolo_raw: list[dict] = []
    detected_seed = False
    refined_any_box = False
    shown_database_seed = False
    identified_label = ""
    identified_conf = 0.0
    h, w = frame.shape[:2]

    for result in results:
        if result.boxes is None:
            continue

        names = result.names or yolo_model.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            yolo_conf = float(box.conf[0])

            if cls_id < 0 or cls_id >= len(class_names):
                continue

            ix1, iy1, ix2, iy2 = [float(v) for v in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = int(ix1 * sx), int(iy1 * sy), int(ix2 * sx), int(iy2 * sy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            box_area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
            if box_area_ratio < MIN_BOX_AREA_RATIO or box_area_ratio > MAX_BOX_AREA_RATIO:
                continue

            # record raw YOLO detection for debugging/visualization
            if DEBUG_SHOW_YOLO and yolo_conf >= YOLO_CONF_THRESHOLD:
                yolo_raw.append({
                    "bbox": (x1, y1, x2, y2),
                    "text": f"YOLO:{names.get(cls_id, str(cls_id))} {yolo_conf:.2f}",
                    "color": (255, 0, 0),
                    "score": float(yolo_conf),
                    "class_conf": float(yolo_conf),
                })

            detected_seed = True
            refined_box = refine_seed_box(frame, (x1, y1, x2, y2))
            if refined_box is not None:
                x1, y1, x2, y2 = refined_box

            yolo_crop = frame[y1:y2, x1:x2]
            if DEBUG_VERBOSE:
                print(f"[DEBUG] YOLO cls={names.get(cls_id, str(cls_id))} id={cls_id} yconf={yolo_conf:.3f} bbox_ratio={box_area_ratio:.6f} bbox=({x1},{y1},{x2},{y2})")
            if cnn_models:
                class_label, class_conf = classify_seed(yolo_crop, cnn_models, class_names)
                class_label = canonical_label(class_label)
                confidence_threshold = CLASS_CONF_THRESHOLD
            else:
                class_label = canonical_label(names.get(cls_id, str(cls_id)))
                class_conf = float(yolo_conf)
                confidence_threshold = YOLO_CONF_THRESHOLD
            class_key = norm_label(class_label)
            # If classifier rejects, optionally save the crop for inspection
            if (class_conf < confidence_threshold) or (class_key not in known_seed_labels):
                if DEBUG_SAVE_FAILED_CROPS:
                    try:
                        global _debug_save_counter
                        _debug_save_counter += 1
                        fname = DEBUG_DIR / f"crop_{int(time.time())}_{_debug_save_counter}_y{int(yolo_conf*1000)}_c{int(class_conf*1000)}.jpg"
                        cv2.imwrite(str(fname), yolo_crop)
                        if DEBUG_VERBOSE:
                            print(f"[DEBUG] Saved failed crop: {fname} (yconf={yolo_conf:.3f}, cconf={class_conf:.3f}, label={class_label})")
                    except Exception as exc:
                        print(f"[DEBUG] Failed saving crop: {exc}")
                continue

            identified_label = class_label
            identified_conf = class_conf
            shown_database_seed = True

            score = float(class_conf) * max(1.0, float(x2 - x1) * float(y2 - y1))
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "text": f"{class_label} {class_conf:.2f}",
                "color": (0, 255, 0),
                "score": score,
                "class_conf": class_conf,
            })

    detections = choose_best_camera_detections(detections)

    if not detected_seed:
        status = ("No seed detected", (0, 0, 255))
    elif shown_database_seed and not detections:
        status = (f"{identified_label} {identified_conf:.2f} - seed found but box needs plain background", (0, 255, 0))
    elif shown_database_seed:
        status = (f"Detected {identified_label} {identified_conf:.2f}", (0, 255, 0))
    else:
        status = ("No reliable database seed match", (0, 215, 255))

    return {
        "detections": detections,
        "yolo_raw": yolo_raw,
        "status": status,
        "updated_at": time.time(),
    }


frame_lock = threading.Lock()
result_lock = threading.Lock()
latest_frame: np.ndarray | None = None
latest_result = {
    "detections": [],
    "status": ("Starting camera...", (0, 215, 255)),
    "updated_at": 0.0,
}
stop_event = threading.Event()


def inference_worker() -> None:
    global latest_frame, latest_result
    while not stop_event.is_set():
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                latest_frame = None

        if frame is None:
            time.sleep(0.005)
            continue

        result = analyze_frame(frame)
        with result_lock:
            latest_result = result


cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

last_submit = 0.0
last_fps_time = time.time()
fps = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from camera.")
        break

    now = time.time()
    if now - last_submit >= INFERENCE_INTERVAL_SECONDS:
        with frame_lock:
            latest_frame = frame.copy()
        last_submit = now

    with result_lock:
        result = {
            "detections": list(latest_result["detections"]),
            "yolo_raw": list(latest_result.get("yolo_raw", [])),
            "status": latest_result["status"],
            "updated_at": latest_result["updated_at"],
        }

    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        draw_detection(frame, x1, y1, x2, y2, det["text"], det["color"])

    # draw raw YOLO detections (debug only)
    for det in result.get("yolo_raw", []):
        x1, y1, x2, y2 = det["bbox"]
        draw_detection(frame, x1, y1, x2, y2, det.get("text", "YOLO"), det.get("color", (255, 0, 0)))

    status_text, status_color = result["status"]
    if status_text:
        draw_status(frame, status_text, status_color)

    debug_lines = [
        f"Detections: {len(result['detections'])}",
        f"YOLO raw: {len(result.get('yolo_raw', []))}",
        f"Status: {status_text}",
    ]
    for det in result["detections"][:3]:
        debug_lines.append(f"{det['text']} | score {det['score']:.2f}")
    draw_debug_overlay(frame, debug_lines)

    frame_count += 1
    if now - last_fps_time >= 1.0:
        fps = frame_count / (now - last_fps_time)
        frame_count = 0
        last_fps_time = now
    cv2.putText(frame, f"Live FPS {fps:.1f}", (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Seed Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stop_event.set()
worker.join(timeout=2.0)
cap.release()
cv2.destroyAllWindows()
