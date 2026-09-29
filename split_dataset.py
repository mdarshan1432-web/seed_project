import os
import shutil
import random

# ===== PATHS =====
train_img_path = "dataset/train/images"
train_lbl_path = "dataset/train/labels"

valid_img_path = "dataset/valid/images"
valid_lbl_path = "dataset/valid/labels"

# ===== CREATE VALID FOLDERS IF NOT EXISTS =====
os.makedirs(valid_img_path, exist_ok=True)
os.makedirs(valid_lbl_path, exist_ok=True)

# ===== GET ALL IMAGES =====
images = [f for f in os.listdir(train_img_path) if f.endswith((".jpg", ".png", ".jpeg"))]

print(f"Total images found: {len(images)}")

# ===== SHUFFLE =====
random.shuffle(images)

# ===== SPLIT 20% =====
split_count = int(0.2 * len(images))
valid_images = images[:split_count]

print(f"Moving {split_count} images to validation...")

# ===== MOVE FILES =====
moved = 0

for img_name in valid_images:
    img_src = os.path.join(train_img_path, img_name)
    img_dst = os.path.join(valid_img_path, img_name)

    label_name = img_name.rsplit(".", 1)[0] + ".txt"
    label_src = os.path.join(train_lbl_path, label_name)
    label_dst = os.path.join(valid_lbl_path, label_name)

    # Move image
    if os.path.exists(img_src):
        shutil.move(img_src, img_dst)

    # Move label
    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)



    moved += 1

print(f"✅ Successfully moved {moved} images + labels to validation")