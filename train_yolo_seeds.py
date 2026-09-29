"""
================================================================================
 train_yolo_seeds.py
 -------------------------------------------------------------------------------
 Fine-tune YOLOv8n on YOUR 54-class seed dataset so that the live-camera
 bounding boxes show actual seed names (Almond, Chilli, Wheat …) instead of
 COCO classes (person, chair, bottle …).

 This script is fully self-contained.  It will:

   1. Read your classification-style dataset
          dataset/
              Almond/   img1.jpg  img2.jpg ...
              Amla/     img1.jpg  ...
              ...
              Wheat/    img1.jpg  ...

   2. Convert it into the YOLOv8 *detection* layout  (one .txt label per
      image, bbox = full image because each crop already shows ONE seed):

          dataset_yolo/
              images/train/<class>__<file>.jpg
              images/val/  <class>__<file>.jpg
              labels/train/<class>__<file>.txt   -> "<class_id> 0.5 0.5 1.0 1.0"
              labels/val/  <class>__<file>.txt

   3. Auto-write data.yaml with 54 class names.

   4. Fine-tune yolov8n.pt for N epochs (default 50) with sensible
      augmentation defaults.

   5. Copy the best weights to  yolov8n_seeds.pt  in this folder.

   6. Print the exact one-line edit you need to make in app.py to start
      using the new weights.

 USAGE  (from inside SEED_PROJECT/):

      pip install ultralytics pillow tqdm
      python train_yolo_seeds.py                 # 50 epochs, 640 img, batch 16
      python train_yolo_seeds.py --epochs 100    # longer training
      python train_yolo_seeds.py --imgsz 416 --batch 8   # for low-RAM laptops
      python train_yolo_seeds.py --skip-prepare  # if dataset_yolo/ already built
      python train_yolo_seeds.py --val-only      # only run mAP evaluation

 NOTES
 -----
 * Each crop is treated as ONE full-frame seed (bbox = entire image).  This
   gives YOLO a strong "what does seed-X look like" signal even though we
   don't have hand-drawn boxes.  After training, the network generalises and
   draws tight boxes on real-world images.
 * 80/20 train/val split, deterministic via SEED.
 * If you later add hand-labelled detection data (LabelImg / Roboflow), drop
   the .txt label files into dataset_yolo/labels/train|val/ and re-run with
   --skip-prepare to fine-tune further.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
HERE                = Path(__file__).resolve().parent
DEFAULT_DATA_DIR    = HERE / "seeds"            # class-folder layout
DEFAULT_YOLO_DIR    = HERE / "dataset_yolo"     # generated detection layout
DEFAULT_BASE_WEIGHT = HERE / "yolov8n.pt"       # pretrained starting point
DEFAULT_OUT_WEIGHT  = HERE / "yolov8n_seeds.pt" # final fine-tuned weights
DEFAULT_RUNS_DIR    = HERE / "runs" / "yolo_seeds"
CLASSES_FILE        = HERE / "classes.txt"
SEED                = 42
VAL_SPLIT           = 0.20
ALLOWED_EXT         = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def banner(msg: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {msg}\n{bar}")


def info(msg: str) -> None:
    print(f"  ›  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    print(f"\n  ✗  FATAL: {msg}\n", file=sys.stderr)
    sys.exit(code)


def _seed_key(label: str) -> str:
    return "_".join(str(label or "").lower().replace("_", " ").split())


def load_classes(data_dir: Path) -> List[str]:
    if CLASSES_FILE.exists():
        classes = [
            ln.strip() for ln in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if classes:
            return classes
        warn("classes.txt is empty, falling back to directory names")

    if not data_dir.is_dir():
        die(f"Dataset directory not found for classes generation: {data_dir}")

    classes = sorted(
        [p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    )
    if not classes:
        die(f"No class folders found in {data_dir}")

    CLASSES_FILE.write_text("\n".join(classes) + "\n", encoding="utf-8")
    info(f"Generated classes.txt with {len(classes)} classes")
    return classes


# --------------------------------------------------------------------------- #
#  dataset preparation
# --------------------------------------------------------------------------- #
def gather_images(data_dir: Path, classes: List[str]) -> List[Tuple[Path, int]]:
    """Walk dataset/<class>/*.<ext> and return [(path, class_id), ...]."""
    items: List[Tuple[Path, int]] = []
    missing: List[str] = []

    for cid, cname in enumerate(classes):
        cdir = data_dir / cname
        if not cdir.is_dir():
            # Try a case-insensitive / whitespace-normalized match
            alt = next(
                (p for p in data_dir.iterdir()
                 if p.is_dir() and _seed_key(p.name) == _seed_key(cname)),
                None,
            )
            if alt is None:
                missing.append(cname)
                continue
            cdir = alt

        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in ALLOWED_EXT:
                items.append((f, cid))

    if missing:
        warn(
            f"{len(missing)} classes had NO folder in {data_dir}: "
            f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}"
        )

    if not items:
        die(f"No images found under {data_dir}.  Expected dataset/<class>/*.jpg")

    return items


def split_train_val(
    items: List[Tuple[Path, int]],
    val_split: float = VAL_SPLIT,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Per-class stratified shuffle so every class shows up in val."""
    rng = random.Random(SEED)
    by_class: dict[int, list] = {}
    for p, c in items:
        by_class.setdefault(c, []).append(p)

    train, val = [], []
    for c, paths in by_class.items():
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_split))) if len(paths) > 1 else 0
        val.extend((p, c) for p in paths[:n_val])
        train.extend((p, c) for p in paths[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_yolo_layout(
    items_train: List[Tuple[Path, int]],
    items_val:   List[Tuple[Path, int]],
    out_dir: Path,
    classes: List[str],
) -> None:
    """
    Copy images into images/{train,val}/, generate one-line label .txt
    file per image (full-image bbox), and write data.yaml.
    """
    if out_dir.exists():
        info(f"Cleaning existing {out_dir} ...")
        shutil.rmtree(out_dir)

    img_train = out_dir / "images" / "train"
    img_val   = out_dir / "images" / "val"
    lbl_train = out_dir / "labels" / "train"
    lbl_val   = out_dir / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_kw):  # type: ignore
            return x

    def _emit(items, img_dir: Path, lbl_dir: Path, tag: str) -> None:
        for src, cid in tqdm(items, desc=f"  copy {tag}", ncols=78):
            cname = classes[cid].replace(" ", "_")
            uniq  = f"{cname}__{src.stem}{src.suffix.lower()}"
            dst_img = img_dir / uniq
            dst_lbl = lbl_dir / (Path(uniq).stem + ".txt")
            try:
                shutil.copy2(src, dst_img)
            except Exception as e:
                warn(f"copy failed for {src}: {e}")
                continue
            # YOLO format: <class> <x_center> <y_center> <w> <h>  (normalised)
            # Full-image bbox -> 0.5 0.5 1.0 1.0
            dst_lbl.write_text(f"{cid} 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    _emit(items_train, img_train, lbl_train, "train")
    _emit(items_val,   img_val,   lbl_val,   "val  ")

    # data.yaml -- absolute paths so YOLO works from any CWD
    yaml_text  = f"path: {out_dir.as_posix()}\n"
    yaml_text += "train: images/train\n"
    yaml_text += "val: images/val\n"
    yaml_text += f"nc: {len(classes)}\n"
    yaml_text += "names:\n"
    for c in classes:
        yaml_text += f"  - {c}\n"
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    info(f"data.yaml written ({len(classes)} classes)")


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
def run_training(
    yolo_dir: Path,
    base_weights: Path,
    out_weights: Path,
    runs_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    patience: int,
    val_only: bool,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        die(
            "ultralytics is not installed.  Run:\n"
            "      pip install ultralytics pillow tqdm"
        )

    if not base_weights.exists():
        warn(f"{base_weights.name} missing -- ultralytics will download it")
    model = YOLO(str(base_weights) if base_weights.exists() else "yolov8n.pt")

    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.exists():
        die(f"data.yaml not found at {data_yaml} -- run without --skip-prepare")

    if val_only:
        banner("Validation only (mAP)")
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
        try:
            info(f"mAP50    : {metrics.box.map50:.4f}")
            info(f"mAP50-95 : {metrics.box.map:.4f}")
        except Exception:
            pass
        return

    banner(f"Training YOLOv8n on seeds  ({epochs} epochs, imgsz={imgsz}, batch={batch})")
    t0 = time.time()
    results = model.train(
        data       = str(data_yaml),
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        device     = device,
        patience   = patience,
        project    = str(runs_dir.parent),
        name       = runs_dir.name,
        exist_ok   = True,
        seed       = SEED,
        # gentle augmentation -- seeds are small / homogeneous
        hsv_h      = 0.015,
        hsv_s      = 0.50,
        hsv_v      = 0.30,
        degrees    = 15.0,
        translate  = 0.10,
        scale      = 0.50,
        fliplr     = 0.50,
        mosaic     = 0.50,
        mixup      = 0.10,
        cos_lr     = True,
        verbose    = True,
    )
    info(f"training finished in {(time.time() - t0) / 60:.1f} min")

    # locate best.pt
    best = None
    save_dir = Path(getattr(results, "save_dir", runs_dir))
    for cand in (
        save_dir / "weights" / "best.pt",
        runs_dir / "weights" / "best.pt",
    ):
        if cand.exists():
            best = cand
            break
    if best is None:
        die("training finished but best.pt was not produced -- check logs")

    shutil.copy2(best, out_weights)
    info(f"copied  {best}  ->  {out_weights.name}")

    # quick val
    try:
        banner("Final validation (mAP)")
        m = YOLO(str(out_weights)).val(data=str(data_yaml), imgsz=imgsz, device=device)
        info(f"mAP50    : {m.box.map50:.4f}")
        info(f"mAP50-95 : {m.box.map:.4f}")
    except Exception as e:
        warn(f"final val failed: {e}")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the 54-class seed dataset.",
    )
    ap.add_argument("--data-dir",     type=Path, default=DEFAULT_DATA_DIR,
                    help=f"class-folder dataset root (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--yolo-dir",     type=Path, default=DEFAULT_YOLO_DIR,
                    help=f"generated YOLO layout (default: {DEFAULT_YOLO_DIR})")
    ap.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHT)
    ap.add_argument("--out-weights",  type=Path, default=DEFAULT_OUT_WEIGHT)
    ap.add_argument("--runs-dir",     type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--imgsz",    type=int, default=640)
    ap.add_argument("--batch",    type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    ap.add_argument("--device", type=str, default="",
                    help="'cpu', '0', '0,1' ... (default: auto)")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="skip dataset conversion (re-use existing dataset_yolo/)")
    ap.add_argument("--val-only", action="store_true",
                    help="skip training, only evaluate existing weights")
    args = ap.parse_args()

    classes = load_classes(args.data_dir)
    banner(f"Seed YOLO fine-tuning  ·  {len(classes)} classes")
    info(f"working dir   : {HERE}")
    info(f"dataset src   : {args.data_dir}")
    info(f"YOLO layout   : {args.yolo_dir}")
    info(f"base weights  : {args.base_weights.name}")
    info(f"out weights   : {args.out_weights.name}")
    info(f"epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}  device={args.device or 'auto'}")

    # ---- 1. prepare ---------------------------------------------------- #
    if args.skip_prepare or args.val_only:
        info("skipping dataset preparation (using existing dataset_yolo/)")
        if not (args.yolo_dir / "data.yaml").exists():
            die(f"--skip-prepare given but {args.yolo_dir}/data.yaml missing")
    else:
        if not args.data_dir.is_dir():
            die(f"dataset folder not found: {args.data_dir}")
        banner("Step 1/2 — Preparing YOLO detection layout from class folders")
        items = gather_images(args.data_dir, classes)
        info(f"total images: {len(items)}")
        train, val = split_train_val(items, args.val_split)
        info(f"split  -> train={len(train)}  val={len(val)}")
        write_yolo_layout(train, val, args.yolo_dir, classes)

    # ---- 2. train ------------------------------------------------------ #
    banner("Step 2/2 — Fine-tuning YOLOv8n")
    run_training(
        yolo_dir     = args.yolo_dir,
        base_weights = args.base_weights,
        out_weights  = args.out_weights,
        runs_dir     = args.runs_dir,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        device       = args.device,
        patience     = args.patience,
        val_only     = args.val_only,
    )

    # ---- 3. one-line patch instructions ------------------------------- #
    if not args.val_only:
        banner("DONE — to use the new weights inside app.py")
        print(f"""
  Open  app.py  and change the YOLO weights path, e.g.:

        # OLD
        YOLO_PATH = BASE_DIR / "yolov8n.pt"
        # NEW
        YOLO_PATH = BASE_DIR / "{args.out_weights.name}"

  Then restart Flask:

        python app.py

  The live camera will now draw boxes labelled with seed names
  (Almond, Chilli, Wheat ...) instead of COCO classes.
""")


if __name__ == "__main__":
    main()
