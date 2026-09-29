import os
import shutil

# SOURCE (your original dataset)
source_folder = "seeds"

# DESTINATION (YOLO folder)
dest_folder = "dataset/train/images"

os.makedirs(dest_folder, exist_ok=True)

count = 0

for class_name in os.listdir(source_folder):
    class_path = os.path.join(source_folder, class_name)

    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            src = os.path.join(class_path, img_name)

            if img_name.endswith((".jpg", ".png", ".jpeg")):
                # rename to avoid duplicate names
                new_name = f"{class_name}_{count}.jpg"
                dst = os.path.join(dest_folder, new_name)

                shutil.copy(src, dst)
                count += 1

print(f"Copied {count} images successfully ✅")