import pathlib, sys, importlib
from PIL import Image
root = pathlib.Path(r'c:/Users/mdars/OneDrive/Desktop/seed_project')
sys.path.insert(0, str(root))
import app as app_module
importlib.reload(app_module)
app_module.app.config['SERVER_NAME'] = 'localhost:5000'

class_samples = []
for class_dir in sorted(root.glob('seeds/*')):
    if class_dir.is_dir():
        imgs = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        if imgs:
            class_samples.append((class_dir.name, imgs[0]))
print('classes sampled', len(class_samples))

mismatches = []
with app_module.app.app_context():
    for true_class, path in class_samples:
        img = Image.open(path).convert('RGB')
        res = app_module._run_full_pipeline(img, path.name, 'upload', False)
        pred = res.get('predicted_class') or ''
        if pred != true_class:
            mismatches.append((true_class, str(path.relative_to(root)), pred, res.get('confidence'), res.get('source')))

print('mismatch count', len(mismatches))
for item in mismatches:
    print(item)
