AgroSeed.AI - Real-Time Agriculture Seeds Detection and Recognition
===================================================================

AgroSeed.AI is a Flask-based web application for detecting and recognizing
agricultural seeds using AI. It supports live camera detection, image upload,
Kannada/English UI labels, Karnataka mandi information, prediction history,
and a controlled online fallback for seeds outside the offline database.


Main Features
-------------

1. Offline seed detection
   - Detects 54 predefined seed classes listed in classes.txt.
   - Uses an ensemble of three CNN models:
     - model1.h5
     - model2.h5
     - model3.h5
   - Works without internet for the supported 54 seed classes.

2. Live camera and upload detection
   - Detect seeds using webcam/mobile camera.
   - Upload seed images from the browser.
   - Supports front and back camera selection on mobile devices.

3. Bounding boxes
   - Uses YOLOv8 when available to draw boxes around detected seed objects.
   - Conservative fallback boxes are used only when a seed-like foreground is
     isolated reliably.
   - It avoids drawing boxes around the full image/frame.

4. Kannada support
   - UI can switch between English and Kannada.
   - Seed names are shown in Kannada where translations are available.

5. Karnataka seed information
   - Shows indicative mandi price, soil type, season, mandis, and notes for
     seeds available in the local knowledge base.

6. Online fallback for seeds outside the 54-class database
   - If a seed is not available in the offline database, the app first shows:
     "Data not available in offline database."
   - If offline, it shows:
     "Data not available. Please connect to the internet."
   - After internet is connected, press Refresh to research external sources.
   - Online results are returned only when confidence is high.
   - If confidence is low, it shows:
     "Low confidence result. Accurate seed data not available."

7. History
   - Predictions are saved in history.csv.
   - History page supports filtering, viewing confidence, and exporting CSV.


Project Files
-------------

app.py
    Main Flask backend. Loads models, handles prediction routes, online
    fallback, history, and APIs.

templates/index.html
    Main web page for live camera, upload, result display, seed catalog, and
    project sections.

templates/history.html
    History page for previous predictions.

static/script.js
    Frontend JavaScript for camera, upload, result rendering, language switch,
    bounding boxes, online refresh, and UI interactions.

static/style.css
    Main styling for the application.

classes.txt
    List of the 54 offline seed classes.

model1.h5, model2.h5, model3.h5
    CNN models used for ensemble prediction.

yolov8n_seeds.pt
    Fine-tuned seed YOLO model file used for object localization/bounding boxes.

history.csv
    Prediction history log. Created/updated automatically.

train_model1.py, train_model2.py, train_model3.py
    Training scripts for CNN models.

train_yolo_seeds.py
    Script for YOLO seed training/fine-tuning.


Requirements
------------

Recommended:
    Python 3.10 or 3.11

Python packages:
    flask
    numpy
    pillow
    ultralytics
    requests
    beautifulsoup4
    werkzeug

Recommended TensorFlow install on Windows:
    pip install tensorflow-directml

If you want native CUDA support on Windows and your GPU drivers are configured,
use a TensorFlow 2.10.x build instead of the latest 2.11+ release:
    pip install "tensorflow==2.10.1"

Install example (CPU-only / compatible):
    pip install flask numpy pillow ultralytics requests beautifulsoup4 werkzeug

Install example (Windows GPU via DirectML):
    pip install flask numpy pillow tensorflow-directml ultralytics requests beautifulsoup4 werkzeug

Note:
    TensorFlow >=2.11 on native Windows does not support GPU. Use WSL2 or
    tensorflow-directml for GPU acceleration, or stay on TensorFlow 2.10.1
    for native CUDA/cuDNN support on Windows.


How to Run
----------

1. Open terminal in the project folder:
    c:\Users\mdars\OneDrive\Desktop\seed_project

2. Run:
    python app.py

3. Open in browser:
    http://127.0.0.1:5000

The backend starts on:
    host = 0.0.0.0
    port = 5000


Mobile Camera Usage
-------------------

1. Start the Flask app.
2. Open the app from your phone using the computer IP address, for example:
    http://YOUR_COMPUTER_IP:5000
3. Allow camera permission in the mobile browser.
4. Choose Back camera or Front camera from the camera dropdown.

Important:
    Camera access may require HTTPS on some mobile browsers unless the site is
    opened through localhost or a trusted local network browser setting.


Offline Detection Behavior
--------------------------

The 54 seed classes in classes.txt are detected offline using the local CNN
ensemble. For these classes, internet is not required.

If the system is confident and the seed exists in the local database, it shows:
    - predicted seed name
    - confidence
    - top-5 alternatives
    - Karnataka price/soil/season/mandi details
    - bounding boxes when available


Online Detection Behavior
-------------------------

Online research is only used for seeds outside the 54 offline classes or for
low-confidence offline predictions.

Flow:
    1. Upload image.
    2. If not available offline, app shows:
       "Data not available in offline database."
    3. Connect to internet if needed.
    4. Press Refresh.
    5. App searches online sources and checks confidence.
    6. If high confidence, it shows:
       - seed name
       - Kannada name if found
       - market price snippets
       - suitable soil snippets
       - other basic details
       - source links
       - confidence percentage
    7. If low confidence, it shows:
       "Low confidence result. Accurate seed data not available."

This avoids forcing incorrect detection for unsupported seeds.


Bounding Box Notes
------------------

Bounding boxes appear only when a seed object is detected or when a conservative
foreground estimate can isolate a seed-like object.

The system avoids drawing a bounding box around the full image because that can
make the detection look more confident than it really is.

For best bounding boxes:
    - Use a plain background.
    - Keep the seed clearly visible.
    - Avoid hands covering the seed.
    - Use good lighting.


Adding More Offline Seed Classes
--------------------------------

1. Add image folders under:
    seeds/<new_seed_class>/

2. Retrain the CNN models:
    python train_model1.py
    python train_model2.py
    python train_model3.py

3. Update classes.txt with the new class name.

4. Restart app.py.

Optional:
    Fine-tune YOLO using train_yolo_seeds.py for better bounding boxes.


History
-------

Prediction records are stored in:
    history.csv

The history page is available at:
    http://127.0.0.1:5000/history

You can export or clear history from the UI.


Troubleshooting
---------------

Problem: App starts but models are not loaded.
Solution:
    Make sure model1.h5, model2.h5, and model3.h5 are present in the project
    root folder.

Problem: Bounding boxes are not visible.
Solution:
    Make sure yolov8n_seeds.pt exists. If YOLO cannot detect a seed object and the
Problem: Online detection does not work.
Solution:
    Check internet connection and install:
        pip install requests beautifulsoup4

Problem: Camera does not open.
Solution:
    Allow camera permission in browser. On phones, try the Back camera or Front
    camera option. Some browsers require HTTPS for camera access.

Problem: Kannada text does not display correctly.
Solution:
    Use a modern browser with Unicode font support.


Important Notes
---------------

- Offline detection is reliable only for the 54 trained seed classes.
- Online detection is conservative and returns a result only when confidence is
  high enough.
- Market prices from the local database and online snippets are indicative, not
  official live trading prices.
- For production use, connect to official mandi/APMC APIs and train/fine-tune
  YOLO on the actual seed dataset.


Author / Project
----------------

AgroSeed.AI
Real-Time Agriculture Seeds Detection and Recognition using AI
BCA Research Project