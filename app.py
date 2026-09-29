"""
================================================================================
  REAL-TIME AGRICULTURE SEEDS DETECTION AND RECOGNITION USING AI
  Flask backend (app.py)
  --------------------------------------------------------------
  Author      : (BCA Research Project)
  Description : Loads trained CNN ensemble (model1.h5 + model2.h5 + model3.h5)
                and YOLOv8 (yolov8n_seeds.pt) for real-time + image based seed
                detection of 54 seed classes. Serves an official looking
                Flask web application with live camera, photo upload,
                Karnataka mandi price lookup, soil & season recommendation,
                history log and a guaranteed web-fallback search so the
                system never returns "unknown".
================================================================================
"""
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------ imports --
import os
import io
import csv
import json
import time
import uuid
import base64
import logging
import datetime
import threading
import traceback
import shutil
import urllib.request
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    redirect,
    url_for,
    Response,
    abort,
)
from werkzeug.utils import secure_filename

# Force CPU-only TensorFlow and suppress CUDA startup warnings when CUDA is not installed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Optional heavy deps – wrapped so the server still starts even when a
# library is missing on the host.  The user's local SEED_PROJECT already
# has TensorFlow + Ultralytics installed.
try:
    import tensorflow as tf
    import tensorflow.keras.models # type: ignore
    TF_AVAILABLE = True
except Exception as _tf_err:
    TF_AVAILABLE = False
    _tf_err_msg = str(_tf_err)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as _yolo_err:
    YOLO_AVAILABLE = False
    _yolo_err_msg = str(_yolo_err)

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except Exception:
    WEB_AVAILABLE = False


# ============================================================================
#                              CONFIGURATION
# ============================================================================

BASE_DIR        = Path(__file__).resolve().parent
STATIC_DIR      = BASE_DIR / "static"
TEMPLATE_DIR    = BASE_DIR / "templates"
UPLOAD_DIR      = STATIC_DIR / "uploads"
DATASET_DIR     = BASE_DIR / "dataset"
RUNS_DIR        = BASE_DIR / "runs"

CLASSES_FILE    = BASE_DIR / "classes.txt"
HISTORY_FILE    = BASE_DIR / "history.csv"
LOG_FILE        = BASE_DIR / "agro_system.log"

MODEL1_PATH     = BASE_DIR / "model1.h5"
MODEL2_PATH     = BASE_DIR / "model2.h5"
MODEL3_PATH     = BASE_DIR / "model3.h5"
TRAINED_YOLO_PATH = BASE_DIR / "yolov8n_seeds.pt"
YOLO_PATH       = TRAINED_YOLO_PATH

ALLOWED_EXT     = {"png", "jpg", "jpeg", "bmp", "webp"}
IMG_SIZE        = (224, 224)            # matches typical CNN input
CONF_THRESHOLD  = 0.52                  # camera: below this -> trigger web fallback
UPLOAD_CONF_THRESHOLD = 0.55            # upload: stronger local seed recognition threshold
UPLOAD_YOLO_MIN_ACCEPT = 0.18            # upload: minimum YOLO confidence for seed-only fallback
NO_SEED_THRESH  = 0.16                  # below this -> "no seed in frame" (stricter with trained model)
YOLO_CONF_THRESHOLD = 0.35              # upload: seed-trained YOLO sensitivity for seed images
UPLOAD_YOLO_CONF_THRESHOLD = 0.18        # upload-specific YOLO confidence threshold for seed-only fallback
UPLOAD_YOLO_FAST_ACCEPT = 0.22           # strong YOLO confidence to accept label immediately on upload
CAMERA_YOLO_CONF_THRESHOLD = 0.32       # camera: slightly lower for noisy phone images
CAMERA_CLASS_THRESHOLD = 0.55           # require solid CNN crop confidence for live display
CAMERA_FALLBACK_THRESHOLD = 0.68        # full-frame fallback when single strong seed detected
MAX_CAMERA_DETECTIONS = 8               # reduce overlays, prioritize accuracy
MAX_HISTORY     = 5000                  # cap rows in history.csv
OUT_OF_DB_MSG   = "Not available in the database. Internet connection required."
OFFLINE_DB_MSG  = "Data not available in offline database."
CONNECT_MSG     = "Data not available. Please connect to the internet."
SEED_NOT_FOUND_MSG = "Seed data not available."
LOW_ONLINE_CONF_MSG = "Low confidence result. Accurate seed data not available."
ONLINE_CONF_THRESHOLD = 0.90

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#  SEED-ONLY GATE
#  yolov8n.pt is the default COCO model (80 classes : person, car, dog…)
#  It is NOT trained on seeds.  We therefore:
#    1) ignore any YOLO box whose label is not in our seed vocabulary, and
#    2) when CNN confidence is below NO_SEED_THRESH, return "no seed"
#       instead of a forced wrong label.
#  Once you fine-tune YOLO on the seed dataset (train_yolo.py), the
#  filter will start passing real seed boxes through automatically.
# ----------------------------------------------------------------------------
SEED_ONLY_MODE  = True

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- logging --
logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("agro")
log.info("=" * 72)
log.info("Booting Real-Time Agriculture Seeds Detection backend")
log.info("=" * 72)


# ============================================================================
#                         54 SEED CLASS LIST (fallback)
# ============================================================================
DEFAULT_CLASSES: List[str] = [
    "Almond", "Amla", "Apple", "Avocado", "Betel_Nut", "Bitter_melon",
    "Black_Plum", "Bottle_gourd_long", "Carrot", "Cashew", "Cauliflower",
    "Chilli", "Chinese_Date", "Coriander_leaves", "Date", "Guava",
    "Hog_Plum", "Hyacinth_bean", "Jackfruit", "Litchi", "Malabar_Spinach",
    "Mango", "Okra", "Olive", "Onion", "Orange", "Palm", "Papaya", "Pea",
    "Pumpkin", "Radish", "Ribbed_gourd", "Sapodilla", "Spinach", "Tamarind",
    "Tomato", "Water_spinach", "ash_gourd", "bitter_gourd",
    "chilli_green_long", "cluster_beans", "corn", "cucumber",
    "cucumber_green_long", "dill_leaves", "ladys_finger", "palak",
    "pole_beans", "ridge_gourd_long", "snake_gourd",
    "snake_gourd_green_long", "sponge_gourd", "string_bean", "wax_gourd",
]


def load_classes() -> List[str]:
    """Read classes.txt; fall back to DEFAULT_CLASSES if file missing."""
    if CLASSES_FILE.exists():
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            classes = [c.strip() for c in f.readlines() if c.strip()]
        if classes:
            log.info("Loaded %d classes from classes.txt", len(classes))
            return classes
    log.warning("classes.txt not found – using built-in 54-class list")
    return DEFAULT_CLASSES[:]


CLASSES: List[str] = load_classes()

# Tokenised seed vocabulary for the YOLO filter — accepts either the exact
# class name OR any whitespace/underscore variant.
def _build_seed_tokens(classes: List[str]) -> set:
    toks = set()
    for c in classes:
        c_low = c.lower().strip()
        toks.add(c_low)
        toks.add(c_low.replace("_", " "))
        toks.add(c_low.replace(" ", "_"))
        for piece in c_low.replace("_", " ").split():
            if len(piece) > 2:
                toks.add(piece)
    # also allow a few generic synonyms a fine-tuned YOLO might emit
    toks.update({"seed", "seeds", "grain", "grains", "kernel", "bean"})
    return toks
SEED_TOKENS = _build_seed_tokens(CLASSES)


def _seed_key(label: str) -> str:
    return "_".join(str(label or "").lower().replace("_", " ").split())


KNOWN_SEED_KEYS = {_seed_key(c) for c in CLASSES}


# ============================================================================
#               KARNATAKA SEED KNOWLEDGE BASE  (price / soil / season)
#   Prices are indicative ₹/kg ranges from popular Karnataka APMC mandis
#   (Bengaluru, Mysuru, Hubballi, Belagavi, Tumakuru, Davanagere,
#    Mangaluru, Kalaburagi).  Update these from the live AGMARKNET feed
#   if you want exact daily figures.
# ============================================================================
KARNATAKA_DB: Dict[str, Dict[str, Any]] = {
    "Almond": {
        "price_inr_per_kg": "₹780 – ₹950",
        "price_inr_per_kg_kn": "₹780 – ₹950",
        "soil": "Deep, well-drained loamy soil, pH 6.5-7.5",
        "soil_kn": "ಗಹಿರ, ಚೆನ್ನಾಗಿ ತೊಳೆದ ಕೂಟೆಯ ಮಣ್ಣು, pH 6.5-7.5",
        "season": "Sown Dec-Feb (cool dry season)",
        "season_kn": "ಡಿಸೆಂಬರ್-ಫೆಬ್ರುವರಿ ಬಿತ್ತೆ (ತಂಪಾದ ಶುಷ್ಕ ಋತು)",
        "mandis": ["Bengaluru APMC", "Hubballi", "Mangaluru"],
        "notes": "Mostly imported; small high-altitude trial plots in Chikkamagaluru.",
        "notes_kn": "ಹೆಚ್ಚಾಗಿ ರಾಷ್ಟ್ರೀಯಕೃತ; ಚಿಕ್ಕಮಗಳೂರುಯಲ್ಲಿ ಅಲ್ಪ ಎತ್ತರದ ಪ್ರಯೋಗ ಕಥೆ.",
    },
    "Amla": {
        "price_inr_per_kg": "₹40 – ₹70",
        "soil": "Sandy loam, tolerates saline/alkaline soils",
        "season": "Aug-Sep planting; harvest Nov-Feb",
        "mandis": ["Tumakuru", "Chitradurga", "Kolar"],
        "notes": "Drought-hardy, widely grown across central Karnataka dry belt.",
    },
    "Apple": {
        "price_inr_per_kg": "₹110 – ₹180",
        "soil": "Well-drained loam, pH 5.5-6.5",
        "season": "Plant Jan-Feb; harvest Jul-Sep (only in Chikkamagaluru hills)",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Limited cultivation; mostly traded from Himachal/Kashmir.",
    },
    "Avocado": {
        "price_inr_per_kg": "₹220 – ₹350",
        "soil": "Sandy loam, very well drained, pH 6-6.5",
        "season": "Plant Jun-Aug; fruits Aug-Mar",
        "mandis": ["Bengaluru APMC", "Mangaluru", "Madikeri (Kodagu)"],
        "notes": "Kodagu and Chikkamagaluru produce premium grade.",
    },
    "Betel_Nut": {
        "price_inr_per_kg": "₹430 – ₹520",
        "soil": "Deep clay loam with high organic content",
        "season": "Plant Jun-Aug (monsoon); harvest 5-7 yrs later, Aug-Nov",
        "mandis": ["Sirsi", "Shivamogga", "Mangaluru", "Sagara"],
        "notes": "Karnataka is India's largest arecanut producer.",
    },
    "Bitter_melon": {
        "price_inr_per_kg": "₹35 – ₹60",
        "soil": "Sandy loam rich in organic matter, pH 6-7",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru", "Davanagere"],
        "notes": "Short duration crop – 55-60 days.",
    },
    "Black_Plum": {
        "price_inr_per_kg": "₹80 – ₹140",
        "soil": "Deep alluvial / loamy soils",
        "season": "Plant Jul-Aug; fruits May-Jun",
        "mandis": ["Bengaluru APMC", "Belagavi"],
        "notes": "Locally called Nerale Hannu; abundant in Western Ghats foothills.",
    },
    "Bottle_gourd_long": {
        "price_inr_per_kg": "₹18 – ₹32",
        "soil": "Sandy loam, pH 6-7, well drained",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Tumakuru", "Hubballi"],
        "notes": "Trellis cultivation increases yield 3x.",
    },
    "Carrot": {
        "price_inr_per_kg": "₹28 – ₹55",
        "soil": "Deep, loose sandy loam, pH 6.5",
        "season": "Aug-Nov (rabi)",
        "mandis": ["Chikkaballapur", "Kolar", "Hassan"],
        "notes": "Karnataka's Ooty-X and Pusa Rudhira do well in Hassan.",
    },
    "Cashew": {
        "price_inr_per_kg": "₹680 – ₹950",
        "soil": "Red sandy loam, lateritic soil",
        "season": "Plant Jun-Jul; harvest Feb-May",
        "mandis": ["Mangaluru", "Karwar", "Puttur", "Kundapur"],
        "notes": "Coastal Karnataka is a major cashew belt.",
    },
    "Cauliflower": {
        "price_inr_per_kg": "₹25 – ₹45",
        "soil": "Loamy soil, pH 6-7, rich in organic matter",
        "season": "Sep-Nov (main); Jan-Feb (late)",
        "mandis": ["Belagavi", "Hubballi", "Mysuru"],
        "notes": "Cool nights (15-20 °C) needed for good curd formation.",
    },
    "Chilli": {
        "price_inr_per_kg": "₹150 – ₹260 (dry)",
        "price_inr_per_kg_kn": "₹150 – ₹260 (ಆರಿದ)",
        "soil": "Black cotton or red loam, pH 6.5-7.5",
        "soil_kn": "ಕಪ್ಪು ಹತ್ತಿ ಅಥವಾ ಕೆಂಪು ದೋವೆ, pH 6.5-7.5",
        "season": "Kharif Jun-Jul; Rabi Sep-Oct",
        "season_kn": "ಖರೀಫ್ ಜೂನ್-ಜುಲೈ; ರಬೀ ಸೆಪ್ಟೆಂಬರ್-ಅಕ್ಟೋಬರ್",
        "mandis": ["Byadagi", "Hubballi", "Raichur", "Ballari"],
        "notes": "Byadagi chilli – GI tagged, exported worldwide.",
        "notes_kn": "ಬ್ಯಾಡಗಿ ಮೆಣಸು - GI ಟ್ಯಾಗ್ ಮಾಡಿದ, ವಿಶ್ವಾದ್ಯಾಪಿ ರಫ್ತಾದ.",
    },
    "Chinese_Date": {
        "price_inr_per_kg": "₹160 – ₹240",
        "soil": "Sandy / sandy loam, drought tolerant",
        "season": "Plant Jul-Aug; harvest Sep-Oct",
        "mandis": ["Bengaluru APMC"],
        "notes": "Jujube (Ber); some orchards in Kolar & Chitradurga.",
    },
    "Coriander_leaves": {
        "price_inr_per_kg": "₹40 – ₹80",
        "soil": "Loam / sandy loam, pH 6.5-7.5",
        "season": "Oct-Dec (rabi); Jun-Jul (kharif)",
        "mandis": ["Bengaluru APMC", "Mysuru", "Davanagere"],
        "notes": "Short duration 30-40 days; high water need.",
    },
    "Date": {
        "price_inr_per_kg": "₹220 – ₹420",
        "soil": "Sandy soil, drought hardy",
        "season": "Plant Feb-Mar; harvest Jul-Sep",
        "mandis": ["Bengaluru APMC", "Kalaburagi"],
        "notes": "Trial plantations in north Karnataka dry zone.",
    },
    "Guava": {
        "price_inr_per_kg": "₹35 – ₹70",
        "soil": "Wide adaptability; loamy soils best, pH 4.5-8.2",
        "season": "Plant Jun-Sep; harvest Aug-Oct & Feb-Mar",
        "mandis": ["Chikkaballapur", "Bengaluru APMC", "Tumakuru"],
        "notes": "Allahabad Safeda and Lucknow-49 popular.",
    },
    "Hog_Plum": {
        "price_inr_per_kg": "₹60 – ₹110",
        "soil": "Loamy / lateritic soils",
        "season": "Plant Jun-Jul; fruits Jul-Sep",
        "mandis": ["Mangaluru", "Udupi"],
        "notes": "Ambazhanga / Amate kayi – coastal favourite.",
    },
    "Hyacinth_bean": {
        "price_inr_per_kg": "₹40 – ₹70",
        "soil": "Loamy / black cotton soil",
        "season": "Kharif Jun-Jul",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Avare kalu – Karnataka winter delicacy.",
    },
    "Jackfruit": {
        "price_inr_per_kg": "₹25 – ₹55 (whole)",
        "soil": "Deep alluvial / lateritic",
        "season": "Plant Jun-Aug; harvest Mar-Jun",
        "mandis": ["Tumakuru (Toobagere)", "Mangaluru", "Shivamogga"],
        "notes": "Tubagere jackfruit GI tagged.",
    },
    "Litchi": {
        "price_inr_per_kg": "₹140 – ₹220",
        "soil": "Deep, well-drained loamy, pH 5.5-7",
        "season": "Plant Jun-Sep; harvest May-Jun",
        "mandis": ["Bengaluru APMC"],
        "notes": "Limited cultivation in Chikkamagaluru hills.",
    },
    "Malabar_Spinach": {
        "price_inr_per_kg": "₹30 – ₹60",
        "soil": "Loamy soil rich in organic matter",
        "season": "Year-round; best Jun-Oct",
        "mandis": ["Mangaluru", "Bengaluru APMC"],
        "notes": "Basale soppu – staple in coastal cuisine.",
    },
    "Mango": {
        "price_inr_per_kg": "₹60 – ₹180",
        "price_inr_per_kg_kn": "₹60 – ₹180",
        "soil": "Wide range; deep alluvial loam best",
        "soil_kn": "ವಿಶಾಲ ಶ್ರೇಣಿ; ಗಹಿರ ಅಲ್ಲುವಿಯಲ್ ದೋವೆ ಸೆರೆಸ",
        "season": "Plant Jun-Sep; harvest Apr-Jul",
        "season_kn": "ಸಿಪ್ಪೆ ಜೂನ್-ಸೆಪ್ಟೆಂಬರ್; ಸಂಗ್ರಹ ಏಪ್ರಿಲ್-ಜುಲೈ",
        "mandis": ["Ramanagara", "Kolar", "Srinivaspur", "Belagavi"],
        "notes": "Karnataka is 3rd largest mango producer; Alphonso, Raspuri, Badami.",
        "notes_kn": "ಕರ್ನಾಟಕ 3 ನೇ ದೊಡ್ಡ ಮಾವಿನ ಉತ್ಪಾದಕ; ಅಲ್ಫೋನ್ಸೋ, ರಸ್ಪುರಿ, ಬದಾಮಿ.",
    },
    "Okra": {
        "price_inr_per_kg": "₹30 – ₹55",
        "soil": "Sandy loam to clay loam, pH 6-6.8",
        "season": "Kharif Jun-Jul; Summer Feb-Mar",
        "mandis": ["Bengaluru APMC", "Hubballi", "Davanagere"],
        "notes": "Bhindi – popular cash crop, 45-50 day cycle.",
    },
    "Olive": {
        "price_inr_per_kg": "₹420 – ₹650",
        "soil": "Well-drained loam, pH 6.5-8.5",
        "season": "Plant Jan-Feb",
        "mandis": ["Bengaluru APMC"],
        "notes": "Mostly imported; experimental plots in Bagalkote.",
    },
    "Onion": {
        "price_inr_per_kg": "₹22 – ₹48",
        "soil": "Sandy loam to clay loam, pH 6-7",
        "season": "Kharif Jun-Jul; Rabi Oct-Nov",
        "mandis": ["Hubballi", "Vijayapura", "Belagavi", "Kalaburagi"],
        "notes": "North Karnataka – major bulb onion belt.",
    },
    "Orange": {
        "price_inr_per_kg": "₹50 – ₹95",
        "soil": "Deep, well-drained sandy loam, pH 6-7.5",
        "season": "Plant Jun-Sep; harvest Nov-Mar",
        "mandis": ["Belagavi", "Bengaluru APMC"],
        "notes": "Coorg mandarin GI; also Belagavi belt.",
    },
    "Palm": {
        "price_inr_per_kg": "₹35 – ₹85",
        "soil": "Sandy / sandy loam, coastal soils",
        "season": "Plant Jun-Aug",
        "mandis": ["Mangaluru", "Karwar"],
        "notes": "Palmyra & oil palm grown in coastal Karnataka.",
    },
    "Papaya": {
        "price_inr_per_kg": "₹18 – ₹35",
        "soil": "Light loam, well-drained, pH 6-7",
        "season": "Plant Feb-Mar / Jun-Jul",
        "mandis": ["Bengaluru APMC", "Tumakuru", "Chitradurga"],
        "notes": "Red Lady, Taiwan-786 hybrids most cultivated.",
    },
    "Pea": {
        "price_inr_per_kg": "₹50 – ₹90",
        "soil": "Loamy / clay loam, pH 6-7.5",
        "season": "Rabi Oct-Nov",
        "mandis": ["Hassan", "Chikkamagaluru", "Belagavi"],
        "notes": "Cool weather crop, 90-100 day cycle.",
    },
    "Pumpkin": {
        "price_inr_per_kg": "₹14 – ₹28",
        "soil": "Loamy soil rich in OM, pH 6-7",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Long shelf life; ideal for storage.",
    },
    "Radish": {
        "price_inr_per_kg": "₹18 – ₹35",
        "soil": "Deep loose sandy loam, pH 6.5",
        "season": "Aug-Feb",
        "mandis": ["Kolar", "Chikkaballapur", "Hassan"],
        "notes": "Pusa Chetki, Pusa Himani common varieties.",
    },
    "Ribbed_gourd": {
        "price_inr_per_kg": "₹25 – ₹40",
        "soil": "Sandy loam, well-drained",
        "season": "Kharif Jun-Jul; Summer Feb-Mar",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Heerekai – staple gourd in south Karnataka.",
    },
    "Sapodilla": {
        "price_inr_per_kg": "₹40 – ₹75",
        "soil": "Deep alluvial / sandy loam",
        "season": "Plant Jun-Aug; fruits Dec-Mar",
        "mandis": ["Bengaluru APMC", "Mangaluru"],
        "notes": "Chikku – Kalipatti and Cricket Ball varieties.",
    },
    "Spinach": {
        "price_inr_per_kg": "₹25 – ₹50",
        "soil": "Loamy / sandy loam, pH 6-7",
        "season": "Sep-Feb (rabi); also year-round under shade",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "All-Green and Pusa Jyoti popular.",
    },
    "Tamarind": {
        "price_inr_per_kg": "₹95 – ₹160",
        "soil": "Wide adaptability; deep alluvial best",
        "season": "Plant Jun-Sep; harvest Feb-Apr",
        "mandis": ["Chitradurga", "Tumakuru", "Hubballi"],
        "notes": "Karnataka 2nd in tamarind production.",
    },
    "Tomato": {
        "price_inr_per_kg": "₹15 – ₹70 (volatile)",
        "price_inr_per_kg_kn": "₹15 – ₹70 (ಬದಲಾಗುವ)",
        "soil": "Sandy loam to clay loam, pH 6-7",
        "soil_kn": "ಮರಳಿನ ದೋವೆಯಿಂದ ಮಣ್ಣು ದೋವೆ, pH 6-7",
        "season": "Kharif Jun-Jul; Rabi Oct-Nov; Summer Jan-Feb",
        "season_kn": "ಖರೀಫ್ ಜೂನ್-ಜುಲೈ; ರಬೀ ಅಕ್ಟೋಬರ್-ನವೆಂಬರ್; ಬೇಸಿಗೆ ಜನವರಿ-ಫೆಬ್ರುವರಿ",
        "mandis": ["Kolar", "Chikkaballapur", "Bengaluru APMC", "Hassan"],
        "notes": "Kolar is Asia's 2nd largest tomato market.",
        "notes_kn": "ಕೋಲಾರ್ ಆಸಿಯಾದ 2 ನೇ ದೊಡ್ಡ ಟೊಮೆಟೊ ಮಾರುಕಟ್ಟೆ.",
    },
    "Water_spinach": {
        "price_inr_per_kg": "₹25 – ₹50",
        "soil": "Marshy, water-logged or moist loamy",
        "season": "Year round; best Jun-Oct",
        "mandis": ["Mangaluru", "Bengaluru APMC"],
        "notes": "Kankun / Nadunni soppu.",
    },
    "ash_gourd": {
        "price_inr_per_kg": "₹14 – ₹26",
        "soil": "Sandy loam, well-drained",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Boodi kumbalakai – long storage.",
    },
    "bitter_gourd": {
        "price_inr_per_kg": "₹35 – ₹60",
        "soil": "Sandy loam rich in OM",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Davanagere"],
        "notes": "Hagalakai; trellis improves yield.",
    },
    "chilli_green_long": {
        "price_inr_per_kg": "₹40 – ₹80",
        "soil": "Black cotton / red loam",
        "season": "Kharif Jun-Jul; Rabi Oct-Nov",
        "mandis": ["Hubballi", "Raichur", "Bengaluru APMC"],
        "notes": "Long green chillies – fresh market.",
    },
    "cluster_beans": {
        "price_inr_per_kg": "₹35 – ₹70",
        "soil": "Sandy loam, drought tolerant",
        "season": "Kharif Jun-Jul",
        "mandis": ["Tumakuru", "Chitradurga"],
        "notes": "Gorikai / Guvar – dryland crop.",
    },
    "corn": {
        "price_inr_per_kg": "₹20 – ₹35",
        "soil": "Well-drained loamy soils, pH 5.5-7",
        "season": "Kharif Jun-Jul; Rabi Oct-Nov",
        "mandis": ["Davanagere", "Haveri", "Chitradurga", "Belagavi"],
        "notes": "Karnataka leads sweet corn & baby corn production.",
    },
    "cucumber": {
        "price_inr_per_kg": "₹18 – ₹35",
        "soil": "Sandy loam, pH 6.5-7",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Southavare / Sowtekai.",
    },
    "cucumber_green_long": {
        "price_inr_per_kg": "₹22 – ₹40",
        "soil": "Sandy loam, well-drained",
        "season": "Year round under polyhouse",
        "mandis": ["Bengaluru APMC"],
        "notes": "Polyhouse / hi-tech cultivation popular near Bengaluru.",
    },
    "dill_leaves": {
        "price_inr_per_kg": "₹35 – ₹70",
        "soil": "Loamy soils, pH 6-7.5",
        "season": "Oct-Feb (rabi)",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Sabbasige soppu – cool season herb.",
    },
    "ladys_finger": {
        "price_inr_per_kg": "₹30 – ₹55",
        "soil": "Sandy loam to clay loam",
        "season": "Kharif Jun-Jul; Summer Feb-Mar",
        "mandis": ["Bengaluru APMC", "Hubballi", "Davanagere"],
        "notes": "Bendekai – another name for Okra.",
    },
    "palak": {
        "price_inr_per_kg": "₹20 – ₹40",
        "soil": "Loamy / sandy loam, pH 6-7",
        "season": "Sep-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Palak Pusa Bharathi performs well.",
    },
    "pole_beans": {
        "price_inr_per_kg": "₹40 – ₹80",
        "soil": "Loamy / red soil, pH 6-7",
        "season": "Kharif Jun-Jul; Rabi Oct-Nov",
        "mandis": ["Hassan", "Chikkamagaluru", "Bengaluru APMC"],
        "notes": "Climber bean; trellised cultivation.",
    },
    "ridge_gourd_long": {
        "price_inr_per_kg": "₹25 – ₹45",
        "soil": "Sandy loam rich in OM",
        "season": "Kharif Jun-Jul; Summer Feb-Mar",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Long Heerekai – higher market price.",
    },
    "snake_gourd": {
        "price_inr_per_kg": "₹25 – ₹40",
        "soil": "Sandy loam, well-drained",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mangaluru"],
        "notes": "Padavalakai – grows up to 1 m long.",
    },
    "snake_gourd_green_long": {
        "price_inr_per_kg": "₹28 – ₹48",
        "soil": "Sandy loam rich in OM",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Premium long variety.",
    },
    "sponge_gourd": {
        "price_inr_per_kg": "₹22 – ₹38",
        "soil": "Sandy loam, well-drained",
        "season": "Kharif Jun-Jul; Summer Jan-Feb",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Tooriai / Sponge gourd.",
    },
    "string_bean": {
        "price_inr_per_kg": "₹35 – ₹65",
        "soil": "Loamy soils, pH 6-7",
        "season": "Kharif Jun-Jul; Rabi Oct-Nov",
        "mandis": ["Hassan", "Bengaluru APMC"],
        "notes": "Long pod beans – fresh market.",
    },
    "wax_gourd": {
        "price_inr_per_kg": "₹16 – ₹30",
        "soil": "Sandy loam, pH 6-7",
        "season": "Kharif Jun-Jul",
        "mandis": ["Bengaluru APMC", "Mysuru"],
        "notes": "Boodi kumbalakai (white pumpkin).",
    },
}


def get_seed_info(name: str) -> Dict[str, Any]:
    """Return Karnataka info for a seed name (case-insensitive lookup).

    The returned dict includes Kannada fallback fields so the frontend can
    render Kannada text even when the database entry only contains English.
    """
    if not name:
        return {}

    def _normalize(entry: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(entry)
        if "price_inr_per_kg_kn" not in data:
            data["price_inr_per_kg_kn"] = data.get("price_inr_per_kg", "")
        if "soil_kn" not in data:
            data["soil_kn"] = data.get("soil", "")
        if "season_kn" not in data:
            data["season_kn"] = data.get("season", "")
        if "notes_kn" not in data:
            data["notes_kn"] = data.get("notes", "")
        return data

    if name in KARNATAKA_DB:
        return _normalize(KARNATAKA_DB[name])

    # case-insensitive / underscore-insensitive match
    key_norm = "_".join(name.lower().replace("_", " ").split())
    for k, v in KARNATAKA_DB.items():
        if "_".join(k.lower().replace("_", " ").split()) == key_norm:
            return _normalize(v)
    return {}


# ============================================================================
#                       MODEL LOADING (ensemble + YOLO)
# ============================================================================

class ModelManager:
    """Holds the three CNN models and the YOLO detector."""

    def __init__(self) -> None:
        self.cnn_models: List[Any] = []
        self.cnn_model_paths: List[str] = []
        self.cnn_input: Tuple[int, int] = IMG_SIZE
        self.yolo_model: Optional[Any] = None
        self.lock = threading.Lock()
        self._load()

    # ........................................................ load .........
    def _load(self) -> None:
        if not TF_AVAILABLE:
            log.warning("TensorFlow unavailable – CNN inference disabled")
        else:
            for p in (MODEL1_PATH, MODEL2_PATH, MODEL3_PATH):
                if p.exists():
                    try:
                        m = _try_load_keras_model(p)
                        self.cnn_models.append(m)
                        self.cnn_model_paths.append(str(p))
                        # infer input size
                        try:
                            _, h, w, _ = m.input_shape
                            self.cnn_input = (int(h), int(w))
                        except Exception:
                            pass
                        # check output classes size if available
                        try:
                            out_shape = getattr(m, 'output_shape', None)
                            n_out = int(out_shape[-1]) if out_shape is not None else None
                        except Exception:
                            n_out = None
                        if n_out is not None and n_out != len(CLASSES):
                            log.warning("CNN model %s predicts %s classes but classes.txt has %d entries",
                                        p.name, n_out, len(CLASSES))
                        log.info("Loaded CNN model: %s  input=%s  outputs=%s",
                                 p.name, self.cnn_input, n_out)
                    except Exception as e:
                        log.error("Failed to load %s : %s", p.name, e)
                else:
                    log.warning("CNN model missing: %s", p)

        if not YOLO_AVAILABLE:
            log.warning("Ultralytics unavailable – YOLO disabled")
        elif YOLO_PATH.exists():
            try:
                self.yolo_model = YOLO(str(YOLO_PATH))
                # Verify the model has the expected classes
                if hasattr(self.yolo_model, 'names') and self.yolo_model.names:
                    model_classes = list(self.yolo_model.names.values())
                    log.info("YOLO model classes (%d): %s", len(model_classes), model_classes[:10])
                    # Check if model classes match our seed classes
                    seed_matches = sum(1 for c in model_classes if _seed_key(c) in KNOWN_SEED_KEYS)
                    log.info("YOLO model has %d/%d seed classes", seed_matches, len(model_classes))
                    # Verbose dump of class mapping for debugging
                    try:
                        log.debug("YOLO names mapping: %s", self.yolo_model.names)
                    except Exception:
                        pass
                log.info("Loaded YOLO model: %s", YOLO_PATH.name)
            except Exception as e:
                log.error("YOLO load failed: %s", e)
        else:
            log.warning("YOLO weights not found: %s", YOLO_PATH)

    # ........................................................ predict ......
    def predict_cnn(self, img: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Average the soft-max outputs of the loaded CNN models (ensemble).
        Returns (best_class, best_confidence, top5).
        Ensures consistent preprocessing for both upload and camera inputs.
        """
        if not self.cnn_models:
            return ("", 0.0, [])

        # Pre-process for the inferred input size - use LANCZOS for consistency
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(self.cnn_input, Image.Resampling.LANCZOS)
        x = np.asarray(img_resized).astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        with self.lock:
            preds = []
            for m in self.cnn_models:
                try:
                    p = m.predict(x, verbose=0)[0]
                    preds.append(np.asarray(p, dtype="float32"))
                except Exception as e:
                    log.error("CNN forward-pass error: %s", e)
            if not preds:
                return ("", 0.0, [])

        # pad / trim to max length
        n_classes = max(len(p) for p in preds)
        normed = []
        for p in preds:
            if len(p) < n_classes:
                p = np.pad(p, (0, n_classes - len(p)))
            normed.append(p[:n_classes])
        avg = np.mean(np.stack(normed, axis=0), axis=0)

        # softmax sanity (in case logits)
        if avg.min() < 0 or avg.sum() > 1.5:
            e = np.exp(avg - avg.max())
            avg = e / e.sum()

        idx_sorted = np.argsort(avg)[::-1]
        top5 = [
            (CLASSES[i] if i < len(CLASSES) else f"class_{i}", float(avg[i]))
            for i in idx_sorted[:5]
        ]
        best_idx = int(idx_sorted[0])
        best_cls = CLASSES[best_idx] if best_idx < len(CLASSES) else f"class_{best_idx}"
        # Debug logging: show ensemble summary and mapping
        try:
            log.debug("CNN ensemble models: %s", self.cnn_model_paths)
            log.debug("CNN ensemble output classes=%d  classes.txt=%d", len(avg), len(CLASSES))
            log.debug("CNN prediction: %s (conf=%.4f)  top5=%s", best_cls, float(avg[best_idx]), top5)
        except Exception:
            pass
        return (best_cls, float(avg[best_idx]), top5)

    # ........................................................ yolo .........
    def predict_yolo(self, img: Image.Image, conf: float = YOLO_CONF_THRESHOLD) -> List[Dict[str, Any]]:
        """Returns a list of dicts with bbox, conf, class for each detection."""
        if self.yolo_model is None:
            return []
        try:
            log.debug("YOLO predict called: path=%s  conf=%.3f  img_size=%s", YOLO_PATH, conf, img.size)
            try:
                log.debug("YOLO model names: %s", getattr(self.yolo_model, 'names', None))
            except Exception:
                pass
            with self.lock:
                # Use explicit inference parameters for consistency across upload and camera.
                # Use a larger image size for larger uploads to improve seed detection quality.
                imgsz = min(max(max(img.size), 640), 896)
                imgsz = int(math.ceil(imgsz / 32.0)) * 32
                results = self.yolo_model.predict(
                    np.asarray(img.convert("RGB")),
                    verbose=False,
                    conf=conf,
                    imgsz=imgsz,
                    iou=0.45,
                    max_det=32,
                    augment=False,  # No augmentation for real-time
                )
            out = []
            if results:
                r = results[0]
                names = r.names
                if r.boxes is not None and r.boxes.xyxy is not None:
                    for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                        cls_id = int(r.boxes.cls.cpu().numpy()[i])
                        conf = float(r.boxes.conf.cpu().numpy()[i])
                        cname = names.get(cls_id, str(cls_id))
                        

                        # ---------------------------------------------
                        #  SEED-ONLY FILTER  (suppress person, car, dog…)
                        # ---------------------------------------------
                        if SEED_ONLY_MODE:
                            cname_low = str(cname).lower().strip()
                            cname_key = _seed_key(cname)
                            if cname_low not in SEED_TOKENS and cname_key not in SEED_TOKENS:
                                # not a seed – ignore this box silently
                                continue
                            
                            # Additional validation: ensure the class name matches known seed classes
                            if cname_key not in KNOWN_SEED_KEYS:
                                log.debug("YOLO detected unknown class: %s (key: %s)", cname, cname_key)
                                continue

                        log.debug("YOLO detection: %s (conf: %.3f) at bbox: %s", cname, conf, box)
                        out.append({
                            "bbox": [float(v) for v in box],
                            "conf": conf,
                            "class": cname,
                        })
            return out
        except Exception as e:
            log.error("YOLO predict error: %s", e)
            return []


MODELS = ModelManager()


# ============================================================================
#                          WEB FALLBACK SEARCH
# ============================================================================

def _duckduckgo_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Small helper used by the web fallback; returns title/url/snippet rows."""
    rows: List[Dict[str, str]] = []
    if not WEB_AVAILABLE:
        return rows
    try:
        url = "https://duckduckgo.com/html/?q=" + requests.utils.quote(query)
        r = requests.get(url, timeout=7, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return rows
        soup = BeautifulSoup(r.text, "html.parser")
        for result in soup.select(".result")[:limit]:
            a = result.select_one("a.result__a")
            if not a:
                continue
            snippet_el = result.select_one(".result__snippet")
            rows.append({
                "title": a.get_text(" ", strip=True),
                "url": a.get("href", ""),
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            })
    except Exception as e:
        log.warning("DuckDuckGo search failed for %r: %s", query, e)
    return rows


def _kannada_wikipedia_title(title: str) -> str:
    """Return Kannada Wikipedia title when available; otherwise empty string."""
    if not WEB_AVAILABLE or not title:
        return ""
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", timeout=6, params={
            "action": "query",
            "format": "json",
            "prop": "langlinks",
            "titles": title,
            "lllang": "kn",
        }, headers={"User-Agent": "AgroSeedBot/1.0"})
        pages = (r.json().get("query") or {}).get("pages") or {}
        for page in pages.values():
            links = page.get("langlinks") or []
            if links:
                return links[0].get("*", "")
    except Exception as e:
        log.warning("Kannada title lookup failed: %s", e)
    return ""


def _first_snippet(rows: List[Dict[str, str]], keywords: Tuple[str, ...]) -> str:
    for row in rows:
        text = row.get("snippet", "")
        low = text.lower()
        if text and any(k in low for k in keywords):
            return text
    return rows[0].get("snippet", "") if rows else ""


def approximate_seed_box(img: Image.Image, label: str, conf: float) -> List[Dict[str, Any]]:
    """
    Fallback visual box for plain-background seed photos when YOLO has no
    seed-specific box. It estimates foreground from border colour, but returns
    no box if the foreground is too small/large to be a reliable seed object.
    """
    try:
        rgb = img.convert("RGB").resize((640, 640))
        arr = np.asarray(rgb).astype(np.int16)
        border = np.concatenate([
            arr[:20].reshape(-1, 3), arr[-20:].reshape(-1, 3),
            arr[:, :20].reshape(-1, 3), arr[:, -20:].reshape(-1, 3),
        ])
        bg = np.median(border, axis=0)
        mask = np.abs(arr - bg).sum(axis=2) > 45
        ys, xs = np.where(mask)
        mask_area = len(xs) / float(mask.size)
        if len(xs) < 120 or mask_area < 0.002 or mask_area > 0.45:
            return []

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        box_area = ((x2 - x1 + 1) * (y2 - y1 + 1)) / float(mask.size)
        if box_area > 0.65:
            return []

        sx, sy = img.size[0] / 640, img.size[1] / 640
        pad = 10
        box = [
            max(0, (x1 - pad) * sx), max(0, (y1 - pad) * sy),
            min(img.size[0], (x2 + pad) * sx), min(img.size[1], (y2 + pad) * sy),
        ]
        return [{"bbox": box, "conf": float(max(conf, .35)), "class": label, "fallback": True}]
    except Exception as e:
        log.warning("approximate seed box failed: %s", e)
        return []


def standardize_frame_input(img: Image.Image, source: str = "upload") -> Image.Image:
    """Standardize frame input for both upload and camera to ensure consistent preprocessing.
    
    Mobile phone cameras may produce:
    - Different orientations (EXIF rotation)
    - Various resolutions (720p to 4K)
    - Compressed or low-quality images
    - Different aspect ratios
    
    This function normalizes all inputs to a standard size for consistent model inference.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Fix EXIF rotation for phone images
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    width, height = img.size
    
    # For camera: ensure minimum dimension for mobile photos
    # For upload: allow smaller preprocessed images while preserving the
    # original capture quality for YOLO upload detection.
    min_dim = 320 if source == "camera" else 128
    max_dim = 640 if source == "camera" else 900
    # Calculate scale to fit within bounds while maintaining aspect ratio
    scale = min(max_dim / width if width else 1.0, max_dim / height if height else 1.0, 1.0)
    
    # Ensure we don't get too small an image
    if scale < 1.0 or width < min_dim or height < min_dim:
        new_width = max(min_dim, int(round(width * scale)))
        new_height = max(min_dim, int(round(height * scale)))
        
        # Use LANCZOS for better quality downsampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        log.debug(
            "Standardized %s frame from %dx%d to %dx%d (scale: %.2f)",
            source,
            width,
            height,
            new_width,
            new_height,
            scale,
        )

    return img


def refine_seed_box(img: Image.Image, bbox: List[float]) -> Optional[List[float]]:
    """Return a tight seed-like foreground box inside a broad YOLO seed box."""
    try:
        if not CV2_AVAILABLE or len(bbox) != 4:
            return None

        rgb = img.convert("RGB")
        width, height = rgb.size
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop_rgb = np.asarray(rgb.crop((x1, y1, x2, y2)))
        crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        crop_float = crop_rgb.astype(np.float32)
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 20 or crop_w < 20:
            return None

        def clean_mask(raw_mask: np.ndarray) -> np.ndarray:
            mask_u8 = raw_mask.astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
            return mask_u8

        def candidate_boxes(mask_u8: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes: List[Tuple[int, int, int, int, float]] = []
            crop_area = float(crop_w * crop_h)
            min_area = max(60.0, crop_area * 0.0003)
            max_area = crop_area * 0.70
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_area or area > max_area:
                    continue
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rw < 5 or rh < 5:
                    continue
                rect_ratio = (rw * rh) / crop_area
                if rect_ratio < 0.0003 or rect_ratio > 0.70:
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

        border_n = max(4, min(crop_h, crop_w) // 18)
        border = np.concatenate([
            crop_float[:border_n].reshape(-1, 3),
            crop_float[-border_n:].reshape(-1, 3),
            crop_float[:, :border_n].reshape(-1, 3),
            crop_float[:, -border_n:].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)

        masks: List[np.ndarray] = []
        diff = np.linalg.norm(crop_float - bg, axis=2)
        masks.append(diff > max(28.0, float(np.std(border)) * 1.15))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, otsu_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.extend([otsu_dark > 0, otsu_light > 0])

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        border_bgr = cv2.cvtColor(border.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2BGR)
        border_hsv = cv2.cvtColor(border_bgr, cv2.COLOR_BGR2HSV)[:, 0, :]
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

        boxes: List[Tuple[int, int, int, int, float]] = []
        for raw in masks:
            area = float(np.mean(raw))
            if 0.0003 <= area <= 0.70:
                boxes.extend(candidate_boxes(clean_mask(raw)))

        if not boxes:
            return None

        rx, ry, rw, rh, _score = max(boxes, key=lambda item: item[4])
        rx1, rx2 = int(rx), int(rx + rw)
        ry1, ry2 = int(ry), int(ry + rh)
        pad = max(6, int(min(rx2 - rx1 + 1, ry2 - ry1 + 1) * 0.12))
        rx1, ry1 = max(0, rx1 - pad), max(0, ry1 - pad)
        rx2, ry2 = min(crop_w - 1, rx2 + pad), min(crop_h - 1, ry2 + pad)

        refined_ratio = ((rx2 - rx1) * (ry2 - ry1)) / float(crop_w * crop_h)
        if refined_ratio < 0.0003 or refined_ratio > 0.70:
            return None

        return [
            float(x1 + rx1),
            float(y1 + ry1),
            float(x1 + rx2),
            float(y1 + ry2),
        ]
    except Exception as e:
        log.warning("refine_seed_box failed: %s", e)
        return None


def seed_region_has_signal(img: Image.Image, bbox: List[float]) -> bool:
    """Reject blank/flat YOLO boxes before asking the CNN to name a seed."""
    try:
        if len(bbox) != 4:
            return False
        rgb = img.convert("RGB")
        width, height = rgb.size
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return False

        crop = np.asarray(rgb.crop((x1, y1, x2, y2))).astype(np.float32)
        if crop.size == 0:
            return False

        gray = crop.mean(axis=2)
        color_std = float(np.mean(np.std(crop, axis=(0, 1))))
        gray_std = float(np.std(gray))
        edge = 0.0
        if gray.shape[0] > 1 and gray.shape[1] > 1:
            edge = float(np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1))))
        if max(color_std, gray_std) < 7.0 and edge < 1.5:
            return False

        h, w = gray.shape[:2]
        border_n = max(3, min(h, w) // 16)
        border = np.concatenate([
            crop[:border_n].reshape(-1, 3),
            crop[-border_n:].reshape(-1, 3),
            crop[:, :border_n].reshape(-1, 3),
            crop[:, -border_n:].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)
        diff = np.linalg.norm(crop - bg, axis=2)
        fg_ratio = float(np.mean(diff > max(18.0, float(np.std(border)) * 1.05)))

        # A plain empty frame has almost no foreground; a full-frame camera view
        # with a seed on a natural background may still pass by texture/edges.
        return fg_ratio >= 0.001 or max(color_std, gray_std) >= 14.0 or edge >= 4.0
    except Exception as e:
        log.warning("seed_region_has_signal failed: %s", e)
        return False


def propose_seed_boxes(img: Image.Image, max_boxes: int = 8) -> List[Dict[str, Any]]:
    """
    Camera fallback when YOLO misses small seeds.

    It looks for compact foreground regions in the full frame, then lets the
    CNN decide the exact seed name. This is intentionally used only as a
    proposal source; weak CNN crops are still rejected later.
    """
    try:
        if not CV2_AVAILABLE:
            return []

        rgb = img.convert("RGB")
        width, height = rgb.size
        work_w = 480
        scale = width / float(work_w) if width > work_w else 1.0
        work_h = max(1, int(height / scale))
        small = rgb.resize((work_w, work_h), Image.Resampling.BILINEAR) if scale != 1.0 else rgb
        arr_rgb = np.asarray(small)
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        h, w = arr_rgb.shape[:2]
        frame_area = float(w * h)
        if frame_area <= 0:
            return []

        masks: List[np.ndarray] = []
        arr_float = arr_rgb.astype(np.float32)

        border_n = max(6, min(h, w) // 18)
        border = np.concatenate([
            arr_float[:border_n].reshape(-1, 3),
            arr_float[-border_n:].reshape(-1, 3),
            arr_float[:, :border_n].reshape(-1, 3),
            arr_float[:, -border_n:].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)
        diff = np.linalg.norm(arr_float - bg, axis=2)
        masks.append(diff > max(26.0, float(np.std(border)) * 1.10))

        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, otsu_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.extend([otsu_dark > 0, otsu_light > 0])

        hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        masks.append((sat > max(35.0, float(np.percentile(sat, 70)))) & (val > 25))

        proposals: List[Tuple[int, int, int, int, float]] = []
        kernel = np.ones((5, 5), np.uint8)
        min_area = max(60.0, frame_area * 0.00012)
        max_area = frame_area * 0.22

        for raw in masks:
            raw_area = float(np.mean(raw))
            if raw_area < 0.00008 or raw_area > 0.60:
                continue
            mask = (raw.astype(np.uint8) * 255)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_area or area > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 8 or bh < 8:
                    continue
                rect_area = float(bw * bh)
                rect_ratio = rect_area / frame_area
                if rect_ratio < 0.00012 or rect_ratio > 0.28:
                    continue
                aspect = max(bw / float(bh), bh / float(bw))
                if aspect > 5.5:
                    continue
                fill = area / max(1.0, rect_area)
                if fill < 0.15:
                    continue
                touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 2) + int(y + bh >= h - 2)
                score = area * max(fill, 0.18) / max(aspect, 1.0) / (1.0 + touches)
                proposals.append((x, y, bw, bh, score))

        if not proposals:
            return []

        proposals.sort(key=lambda item: item[4], reverse=True)
        chosen: List[Tuple[int, int, int, int, float]] = []

        def iou(a: Tuple[int, int, int, int, float], b: Tuple[int, int, int, int, float]) -> float:
            ax1, ay1, aw, ah, _ = a
            bx1, by1, bw, bh, _ = b
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = aw * ah + bw * bh - inter
            return inter / float(union) if union else 0.0

        for prop in proposals:
            if all(iou(prop, old) < 0.45 for old in chosen):
                chosen.append(prop)
            if len(chosen) >= max_boxes:
                break

        out: List[Dict[str, Any]] = []
        for x, y, bw, bh, score in chosen:
            pad = max(5, int(min(bw, bh) * 0.18))
            x1 = max(0, x - pad) * scale
            y1 = max(0, y - pad) * scale
            x2 = min(w - 1, x + bw + pad) * scale
            y2 = min(h - 1, y + bh + pad) * scale
            bbox = [float(x1), float(y1), float(min(width, x2)), float(min(height, y2))]
            if seed_region_has_signal(img, bbox):
                out.append({
                    "bbox": bbox,
                    "conf": float(min(0.75, max(0.20, score / max(1.0, frame_area * 0.02)))),
                    "class": "seed",
                    "proposal": True,
                })
        return out
    except Exception as e:
        log.warning("propose_seed_boxes failed: %s", e)
        return []


def _try_load_keras_model(path: Path):
    """Try loading a Keras .h5 model. If load_model fails due to legacy
    config keys (e.g. 'batch_shape'), attempt to reconstruct the model by
    sanitizing the saved JSON config and loading weights manually.
    Returns the model or raises the original exception.
    """
    try:
        return tensorflow.keras.models.load_model(str(path), compile=False)
    except Exception as e:
        msg = repr(e)
        # attempt HDF5 repair for legacy 'batch_shape' in model_config
        try:
            import h5py, json
            with h5py.File(str(path), 'r') as hf:
                if 'model_config' not in hf.attrs:
                    raise
                cfg_text = hf.attrs['model_config']
                cfg = json.loads(cfg_text)

            def _strip_keys(obj):
                # remove or normalize legacy config keys and simplify dtype policy objects
                if isinstance(obj, dict):
                    # handle DTypePolicy objects produced by older Keras
                    if obj.get('class_name') == 'DTypePolicy' and isinstance(obj.get('config'), dict):
                        return obj['config'].get('name', 'float32')
                    out = {}
                    for k, v in obj.items():
                        if k == 'batch_shape':
                            out['batch_input_shape'] = _strip_keys(v)
                        elif k in ('build_input_shape', 'synchronized', 'virtual_batch_size'):
                            continue
                        else:
                            out[k] = _strip_keys(v)
                    return out
                if isinstance(obj, list):
                    return [_strip_keys(v) for v in obj]
                return obj

            # ensure legacy batch keys are removed in-place
            def _remove_batch_keys(o):
                if isinstance(o, dict):
                    for k in list(o.keys()):
                        if k == 'batch_shape':
                            o['batch_input_shape'] = o[k]
                            del o[k]
                        elif k in ('build_input_shape', 'synchronized', 'virtual_batch_size'):
                            del o[k]
                        else:
                            _remove_batch_keys(o[k])
                elif isinstance(o, list):
                    for v in o:
                        _remove_batch_keys(v)

            _remove_batch_keys(cfg)
            # also perform sanitized transform for DTypePolicy entries
            cleaned = _strip_keys(cfg)
            json_text = json.dumps(cleaned)
            # Provide a dummy DTypePolicy to satisfy older Keras configs

            class _DummyDTypePolicy:
                def __init__(self, name='float32'):
                    self.name = name

                @classmethod
                def from_config(cls, config):
                    if isinstance(config, str):
                        return cls(config)
                    if isinstance(config, dict):
                        return cls(config.get('name', 'float32'))
                    return cls('float32')

                @classmethod
                def deserialize(cls, config):
                    return cls.from_config(config)

                def get_config(self):
                    return {'name': self.name}

            m = tf.keras.models.model_from_json(json_text, custom_objects={'DTypePolicy': _DummyDTypePolicy})
            # load weights from the HDF5 file
            m.load_weights(str(path))
            log.info("Recovered model from %s by sanitizing model_config", path.name)
            return m
        except Exception:
            # Re-raise original error to preserve context
            raise e


def propose_seed_boxes_aggressive(img: Image.Image, max_boxes: int = 12) -> List[Dict[str, Any]]:
    """A more permissive proposal generator for upload (phone) images.
    Tries multiple simple thresholding strategies to capture small seeds.
    """
    try:
        if not CV2_AVAILABLE:
            return []
        rgb = img.convert("RGB")
        width, height = rgb.size
        work_w = 640
        scale = width / float(work_w) if width > work_w else 1.0
        work_h = max(1, int(height / scale))
        small = rgb.resize((work_w, work_h), Image.Resampling.BILINEAR) if scale != 1.0 else rgb
        arr = np.asarray(small)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        proposals = []

        # Try adaptive threshold and morphological ops
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        frame_area = float(w * h)
        min_area = max(20.0, frame_area * 0.00002)
        max_area = frame_area * 0.5
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = max(3, int(min(bw, bh) * 0.12))
            x1 = max(0, x - pad) * scale
            y1 = max(0, y - pad) * scale
            x2 = min(w - 1, x + bw + pad) * scale
            y2 = min(h - 1, y + bh + pad) * scale
            proposals.append({"bbox": [float(x1), float(y1), float(min(width, x2)), float(min(height, y2))],
                              "conf": float(min(0.6, max(0.18, area / max(1.0, frame_area * 0.01)))),
                              "class": "seed", "proposal": True})

        # Also try Canny edges -> dilate -> contours
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area * 0.5 or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = max(3, int(min(bw, bh) * 0.12))
            x1 = max(0, x - pad) * scale
            y1 = max(0, y - pad) * scale
            x2 = min(w - 1, x + bw + pad) * scale
            y2 = min(h - 1, y + bh + pad) * scale
            proposals.append({"bbox": [float(x1), float(y1), float(min(width, x2)), float(min(height, y2))],
                              "conf": float(min(0.5, max(0.15, area / max(1.0, frame_area * 0.02)))),
                              "class": "seed", "proposal": True})

        # dedupe by IoU
        chosen = []
        for p in sorted(proposals, key=lambda x: x.get('conf', 0.0), reverse=True):
            keep = True
            for c in chosen:
                if bbox_iou(p['bbox'], c['bbox']) > 0.45:
                    keep = False
                    break
            if keep:
                chosen.append(p)
            if len(chosen) >= max_boxes:
                break
        return chosen
    except Exception as e:
        log.debug('propose_seed_boxes_aggressive failed: %s', e)
        return []


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def det_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ab = a.get("bbox") or []
    bb = b.get("bbox") or []
    if len(ab) != 4 or len(bb) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in ab]
    bx1, by1, bx2, by2 = [float(v) for v in bb]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def apply_nms(dets: List[Dict[str, Any]], iou_threshold: float = 0.40) -> List[Dict[str, Any]]:
    """Remove overlapping bounding boxes with strict NMS for accuracy."""
    if not dets:
        return []
    # Sort by CNN class confidence first (more reliable), then YOLO confidence
    sorted_dets = sorted(
        dets,
        key=lambda item: (
            float(item.get("class_conf", item.get("conf", 0.0))),
            float(item.get("conf", 0.0))
        ),
        reverse=True
    )
    kept: List[Dict[str, Any]] = []
    for det in sorted_dets:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        # Skip if overlaps with any kept detection
        overlaps = [old for old in kept if bbox_iou(bbox, old.get("bbox", [])) > iou_threshold]
        if not overlaps:
            kept.append(det)
        elif float(det.get("class_conf", 0.0)) > max(float(o.get("class_conf", 0.0)) for o in overlaps):
            # Replace if this has higher confidence
            kept = [d for d in kept if d not in overlaps]
            kept.append(det)
    return kept


def choose_best_camera_detection(
    dets: List[Dict[str, Any]],
    frame_label: str,
    frame_conf: float,
) -> List[Dict[str, Any]]:
    """Filter and select best non-overlapping detections for camera display."""
    if len(dets) <= 1:
        return dets

    kept: List[Dict[str, Any]] = []
    for det in sorted(
        dets,
        key=lambda item: (
            float(item.get("class_conf", 0.0)),
            float(item.get("conf", 0.0)),
            float(item.get("det_conf", 0.0)),
        ),
        reverse=True,
    ):
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        has_high_overlap = False
        for old in kept:
            if bbox_iou(bbox, old.get("bbox", [])) > 0.6:
                has_high_overlap = True
                break
        
        if not has_high_overlap:
            kept.append(det)
        elif frame_label and frame_conf >= CAMERA_CLASS_THRESHOLD:
            if all(bbox_iou(bbox, old.get("bbox", [])) > 0.85 for old in kept):
                if float(det.get("class_conf", 0.0)) > float(kept[0].get("class_conf", 0.0)):
                    kept = [det]
                    break
    return kept[:MAX_CAMERA_DETECTIONS]


def unify_camera_seed_labels(
    dets: List[Dict[str, Any]],
    frame_label: str,
    frame_conf: float,
) -> None:
    if len(dets) < 2 or not frame_label or frame_conf < CAMERA_CLASS_THRESHOLD:
        return

    frame_key = _seed_key(frame_label)
    if frame_key not in KNOWN_SEED_KEYS:
        return

    keys = [
        _seed_key(str(d.get("seed_class") or d.get("class") or ""))
        for d in dets
    ]
    counts = Counter(k for k in keys if k)
    if not counts:
        return

    top_key, top_count = counts.most_common(1)[0]
    all_same = len(counts) == 1
    strong_frame = frame_conf >= CAMERA_FALLBACK_THRESHOLD
    majority = top_count >= max(2, math.ceil(len(dets) * 0.75))

    if not strong_frame and not majority and not all_same:
        return

    if frame_key == top_key or (strong_frame and majority):
        unified_label = frame_label
    else:
        unified_label = next(
            (d.get("class") for d in dets if _seed_key(str(d.get("seed_class") or d.get("class") or "")) == top_key),
            frame_label,
        )

    for det in dets:
        det["class"] = unified_label
        det["seed_class"] = unified_label
        det["class_conf"] = max(float(det.get("class_conf", 0.0)), frame_conf if _seed_key(unified_label) == frame_key else 0.0)
        det["conf"] = max(float(det.get("conf", 0.0)), float(det["class_conf"]))
        det["fallback"] = True


def classify_yolo_seed_crops(
    img: Image.Image,
    dets: List[Dict[str, Any]],
    min_class_conf: float = NO_SEED_THRESH,
    min_yolo_conf: float = YOLO_CONF_THRESHOLD,
    frame_label: str = "",
    frame_conf: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, float, List[Tuple[str, float]]]]]:
    """
    Classify every seed box and return only boxes with a reliable seed name.

    YOLO is responsible only for saying "this region is a seed"; the CNN
    ensemble is responsible for the exact crop/seed name. Each overlay label
    uses the CNN crop confidence, not YOLO's generic object confidence.
    """
    if not dets:
        return dets, None

    if not MODELS.cnn_models:
        # No CNN models are available; use YOLO class labels directly where possible.
        fallback: List[Dict[str, Any]] = []
        for det in sorted(dets, key=lambda item: float(item.get("conf", 0.0)), reverse=True):
            if float(det.get("conf", 0.0)) < min_yolo_conf and not det.get("proposal"):
                continue
            lab = det.get("class") or det.get("label") or det.get("name")
            if not lab or _seed_key(lab) not in KNOWN_SEED_KEYS:
                continue
            det2 = dict(det)
            det2["seed_class"] = lab
            det2["class_conf"] = float(det.get("conf", 0.0))
            det2["class"] = lab
            det2["det_conf"] = float(det.get("conf", 0.0))
            det2["conf"] = float(det.get("conf", 0.0))
            det2["refined"] = False
            fallback.append(det2)
            if len(fallback) >= MAX_CAMERA_DETECTIONS:
                break
        if fallback:
            top_label = fallback[0]["class"]
            top_conf = float(fallback[0]["conf"])
            return fallback, (top_label, top_conf, [(top_label, top_conf)])
        return fallback, None

    rgb = img.convert("RGB")
    width, height = rgb.size
    updated: List[Dict[str, Any]] = []
    best: Optional[Tuple[str, float, List[Tuple[str, float]]]] = None
    best_score = -1.0

    # Keep the crop confidence threshold consistent for camera frames.
    # Do not lower it simply because a frame-level label exists.
    for det in sorted(dets, key=lambda item: float(item.get("conf", 0.0)), reverse=True):
        if float(det.get("conf", 0.0)) < min_yolo_conf and not det.get("proposal"):
            continue

        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w < 8 or box_h < 8:
            continue

        pad = max(2.0, min(box_w, box_h) * 0.02)
        crop_box = (
            int(max(0, x1 - pad)),
            int(max(0, y1 - pad)),
            int(min(width, x2 + pad)),
            int(min(height, y2 + pad)),
        )
        crop = rgb.crop(crop_box)
        cls, conf, top5 = MODELS.predict_cnn(crop)

        det2 = dict(det)
        det2["seed_class"] = cls
        det2["class_conf"] = float(conf)
        det2["class"] = cls
        det2["det_conf"] = float(det.get("conf", 0.0))
        det2["conf"] = float(conf)

        if det2["class_conf"] < min_class_conf:
            continue

        refined = refine_seed_box(img, [x1, y1, x2, y2])
        if refined:
            refined_iou = det_iou({"bbox": [x1, y1, x2, y2]}, {"bbox": refined})
            refined_area = max(0.0, (refined[2] - refined[0]) * (refined[3] - refined[1]))
            orig_area = box_w * box_h
            if refined_iou >= 0.35 and 0.20 <= refined_area / max(orig_area, 1.0) <= 1.5:
                det2["bbox"] = refined
                det2["refined"] = True
            else:
                det2["bbox"] = [x1, y1, x2, y2]
                det2["refined"] = False
        else:
            det2["bbox"] = [x1, y1, x2, y2]
            det2["refined"] = False

        det2["bbox_area"] = box_w * box_h
        det2["bbox_area_ratio"] = det2["bbox_area"] / float(width * height)
        area_ratio = min(det2["bbox_area_ratio"], 0.95)
        area_penalty = max(0.05, (1.0 - area_ratio) ** 2)
        score = float(det2.get("class_conf", 0.0)) * max(float(det2.get("det_conf", 0.0)), 0.001) * area_penalty
        det2["crop_score"] = score
        if score > best_score:
            best_score = score
            best = (cls, conf, top5)
        updated.append(det2)
        if len(updated) >= MAX_CAMERA_DETECTIONS * 2:
            break

    updated.sort(
        key=lambda item: (
            float(item.get("crop_score", item.get("conf", 0.0))),
            float(item.get("conf", 0.0)),
            float(item.get("det_conf", 0.0)),
            -float(item.get("bbox_area", 0.0)),
            0 if item.get("proposal") else 1,
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    for det in updated:
        if all(det_iou(det, old) < 0.6 for old in kept):
            kept.append(det)
        if len(kept) >= MAX_CAMERA_DETECTIONS:
            break

    if frame_label and frame_conf >= CAMERA_CLASS_THRESHOLD:
        unify_camera_seed_labels(kept, frame_label, frame_conf)

    return kept, best


def label_seed_boxes_from_frame_prediction(
    img: Image.Image,
    dets: List[Dict[str, Any]],
    label: str,
    conf: float,
) -> List[Dict[str, Any]]:
    """
    Last-resort live-camera overlay.

    If YOLO sees seed objects but crop classification is unstable, use the
    strong full-frame seed prediction to label the localized seed boxes.
    """
    if not dets or not label or conf < CAMERA_CLASS_THRESHOLD:
        return []
    if _seed_key(label) not in KNOWN_SEED_KEYS:
        return []

    out: List[Dict[str, Any]] = []
    width, height = img.size
    img_area = float(width * height)
    for det in sorted(dets, key=lambda item: float(item.get("conf", 0.0)), reverse=True):
        if float(det.get("conf", 0.0)) < CAMERA_YOLO_CONF_THRESHOLD:
            continue
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box_area = max(0.0, (x2 - x1) * (y2 - y1))
        if box_area < img_area * 0.003 or box_area > img_area * 1.02:
            continue
        if not seed_region_has_signal(img, bbox):
            continue

        det2 = dict(det)
        refined = refine_seed_box(img, bbox)
        if refined:
            det2["bbox"] = refined
            det2["refined"] = True
        else:
            det2["refined"] = False
        det2["class"] = label
        det2["seed_class"] = label
        det2["class_conf"] = float(conf)
        det2["det_conf"] = float(det.get("conf", 0.0))
        det2["conf"] = float(conf)
        det2["fallback"] = True
        out.append(det2)
        if len(out) >= MAX_CAMERA_DETECTIONS:
            break
    return out


def image_descriptor(img: Image.Image) -> np.ndarray:
    """Compact colour/texture descriptor for conservative online image matching."""
    small = ImageOps.fit(img.convert("RGB"), (160, 160))
    arr = np.asarray(small).astype(np.float32) / 255.0
    hist_parts = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 1), density=True)
        hist_parts.append(hist)
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    mean = arr.reshape(-1, 3).mean(axis=0)
    std = arr.reshape(-1, 3).std(axis=0)
    vec = np.concatenate(hist_parts + [mean, std, np.array([gx, gy])]).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def descriptor_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(a, b))))


def fetch_reference_image_similarity(uploaded: Image.Image, image_url: str) -> float:
    """Compare upload with a trusted page thumbnail. Returns 0 when unavailable."""
    if not WEB_AVAILABLE or not image_url:
        return 0.0
    try:
        r = requests.get(image_url, timeout=7, headers={"User-Agent": "AgroSeedBot/1.0"})
        if r.status_code != 200 or not r.content:
            return 0.0
        ref = Image.open(io.BytesIO(r.content))
        return descriptor_similarity(image_descriptor(uploaded), image_descriptor(ref))
    except Exception as e:
        log.warning("reference image compare failed: %s", e)
        return 0.0


def score_online_candidate(uploaded: Image.Image,
                           info: Dict[str, Any],
                           local_conf: float) -> float:
    """
    Conservative confidence score. It is not a lab-grade accuracy guarantee;
    it only lets online results through when several independent signals agree.
    """
    query = str(info.get("query") or "").replace("_", " ").lower().strip()
    seed_name = str(info.get("seed_name") or "").lower().strip()
    summary = str(info.get("summary") or "").lower()
    details = str(info.get("basic_details") or "").lower()
    links = info.get("links") or []

    source_score = min(1.0, len([l for l in links if l.get("url")]) / 3.0)
    text = f"{seed_name} {summary} {details}"
    seed_terms = sum(1 for term in ("seed", "plant", "crop", "cultivation", "soil") if term in text)
    text_score = min(1.0, seed_terms / 4.0)
    name_score = 1.0 if query and (query in seed_name or query in text) else 0.35
    visual_score = fetch_reference_image_similarity(uploaded, str(info.get("image_url") or ""))

    # Require web evidence to dominate. Local confidence is only a weak hint.
    score = (
        0.28 * source_score
        + 0.24 * text_score
        + 0.18 * name_score
        + 0.20 * visual_score
        + 0.10 * min(1.0, max(0.0, local_conf))
    )
    return round(float(score), 4)


def web_fallback(query: str) -> Dict[str, Any]:
    """
    When the model confidence is low or class is unknown, scrape the open
    web (Wikipedia summary + DuckDuckGo HTML) so the API never returns
    "not detected".  Returns a normalised dict.
    """
    info: Dict[str, Any] = {
        "source": "web_fallback",
        "query": query,
        "summary": "",
        "image_url": "",
        "links": [],
        "seed_name": query.replace("_", " ").strip(),
        "seed_name_kn": "",
        "market_price": "",
        "soil": "",
        "basic_details": "",
        "requires_internet": True,
        "message": OUT_OF_DB_MSG,
        "online_success": False,
        "seed_data_available": False,
        "online_confidence": 0.0,
        "confidence_explanation": "",
    }
    if not WEB_AVAILABLE:
        info["summary"] = (
            f"Identified visually as '{query}'. Network search disabled "
            "on this host – please install requests + beautifulsoup4 to "
            "enable live web fallback."
        )
        return info

    # ---------------------------------------------------- Wikipedia REST ---
    try:
        wiki_q = query.replace("_", " ").strip()
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + requests.utils.quote(wiki_q))
        r = requests.get(url, timeout=6,
                         headers={"User-Agent": "AgroSeedBot/1.0"})
        if r.status_code == 200:
            data = r.json()
            title = data.get("title", wiki_q)
            info["seed_name"] = title
            info["seed_name_kn"] = _kannada_wikipedia_title(title)
            info["summary"] = data.get("extract", "")
            info["basic_details"] = info["summary"]
            info["image_url"] = (data.get("thumbnail") or {}).get("source", "")
            info["links"].append({
                "title": title,
                "url": data.get("content_urls", {})
                            .get("desktop", {}).get("page", ""),
            })
    except Exception as e:
        log.warning("Wikipedia fallback failed: %s", e)

    # ---------------------------------------------------- DuckDuckGo HTML --
    rows = _duckduckgo_results(f"{query} seed agriculture Karnataka", limit=5)
    price_rows = _duckduckgo_results(f"{query} current market price Karnataka seed", limit=3)
    soil_rows = _duckduckgo_results(f"{query} suitable soil cultivation", limit=3)

    for row in rows:
        info["links"].append({"title": row["title"], "url": row["url"]})

    info["market_price"] = _first_snippet(
        price_rows, ("₹", "rs", "inr", "price", "market", "mandi", "rate")
    )
    info["soil"] = _first_snippet(
        soil_rows, ("soil", "loam", "sandy", "clay", "ph", "drained", "cultivation")
    )
    if not info["basic_details"]:
        info["basic_details"] = _first_snippet(rows, ("seed", "plant", "crop", "cultivation"))

    if not info["summary"]:
        info["summary"] = (
            f"'{query}' seed identified. No Wikipedia summary available "
            "right now, but search links are listed below."
        )
    useful_data = bool(
        query and query.lower().strip() != "unknown seed"
        and (info["basic_details"] or info["market_price"] or info["soil"] or info["links"])
    )
    info["online_success"] = useful_data
    info["seed_data_available"] = useful_data
    if not useful_data:
        info["summary"] = SEED_NOT_FOUND_MSG
    return info


def online_research_seed(img: Image.Image,
                         candidates: List[Tuple[str, float]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Research several candidate names and return the highest-confidence web result."""
    seen = set()
    scored: List[Dict[str, Any]] = []
    best_info: Dict[str, Any] = {}
    best_score = 0.0
    for name, local_conf in candidates:
        clean = (name or "").replace("_", " ").strip()
        key = clean.lower()
        if not clean or key in seen or key == "unknown seed":
            continue
        seen.add(key)
        info = web_fallback(clean)
        online_conf = score_online_candidate(img, info, local_conf)
        info["online_confidence"] = online_conf
        info["confidence_explanation"] = (
            "Score combines trusted source count, seed-related text, candidate-name "
            "agreement, online thumbnail image similarity and local model hint."
        )
        scored.append({
            "class": info.get("seed_name") or clean,
            "confidence": online_conf,
            "source": "online",
        })
        if info.get("seed_data_available") and online_conf > best_score:
            best_score = online_conf
            best_info = info

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    if best_info and best_score >= ONLINE_CONF_THRESHOLD:
        return best_info, scored[:5]

    best: Dict[str, Any] = {
        "source": "web_fallback",
        "summary": LOW_ONLINE_CONF_MSG if scored else SEED_NOT_FOUND_MSG,
        "links": [],
        "seed_data_available": False,
        "online_success": False,
        "online_confidence": scored[0]["confidence"] if scored else 0.0,
        "confidence_explanation": "No online candidate passed the 90% confidence gate.",
    }
    return best, scored[:5]


# ============================================================================
#                       HISTORY  (CSV log of predictions)
# ============================================================================

HISTORY_HEADERS = [
    "id", "timestamp", "filename", "predicted_class",
    "confidence", "source", "top5", "notes",
]


def append_history(row: Dict[str, Any]) -> None:
    """Append a prediction row to history.csv, rotating if oversize."""
    write_header = not HISTORY_FILE.exists()
    try:
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_HEADERS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in HISTORY_HEADERS})
    except Exception as e:
        log.error("History append failed: %s", e)

    # rotate if too large
    try:
        rows = read_history()
        if len(rows) > MAX_HISTORY:
            rows = rows[-MAX_HISTORY:]
            with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=HISTORY_HEADERS)
                w.writeheader()
                w.writerows(rows)
    except Exception as e:
        log.warning("History rotation failed: %s", e)


def read_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        log.error("History read failed: %s", e)
        return []


def clear_history() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    log.info("History cleared")


# ============================================================================
#                       IMAGE HELPERS
# ============================================================================

def allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def save_upload(file_storage) -> Path:
    fname = secure_filename(file_storage.filename or "upload.jpg")
    if not allowed_file(fname):
        raise ValueError("File type not allowed")
    unique = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{fname}"
    dest = UPLOAD_DIR / unique
    file_storage.save(str(dest))
    return dest


def decode_base64_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def save_pil(img: Image.Image, ext: str = "jpg") -> Path:
    unique = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"
    dest = UPLOAD_DIR / unique
    img.convert("RGB").save(str(dest))
    return dest


# ============================================================================
#                                 FLASK APP
# ============================================================================

app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024     # 16 MB upload cap
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0             # avoid stale JS/CSS while developing
# Ensure Flask JSON responses preserve Unicode (don't escape non-ASCII)
app.config["JSON_AS_ASCII"] = False


# Ensure all HTML responses explicitly declare UTF-8 charset
@app.after_request
def set_utf8_charset(response):
    content_type = response.headers.get("Content-Type", "")
    if content_type and "charset" not in content_type.lower():
        if response.mimetype == 'text/html' or response.mimetype == 'application/json':
            response.headers["Content-Type"] = f"{response.mimetype}; charset=utf-8"
    return response


# ----------------------------------------------------------- core routes ---
@app.route("/")
def index():
    """Landing page with live camera + photo upload."""
    return render_template(
        "index.html",
        classes=CLASSES,
        n_classes=len(CLASSES),
        karnataka_db=KARNATAKA_DB,
    )


@app.route("/history")
def history_page():
    rows = read_history()[::-1]   # newest first
    return render_template("history.html", rows=rows, count=len(rows))


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


# --------------------------------------------------- prediction endpoints --
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either a multipart 'file' upload OR JSON {"image": "data:..."}.
    Always returns a result – falls back to web search on low confidence.
    """
    try:
        img: Optional[Image.Image] = None
        saved_path: Optional[Path] = None
        filename: str = ""
        force_online = False

        if "file" in request.files:
            force_online = str(request.form.get("force_online", "")).lower() in {"1", "true", "yes"}
            f = request.files["file"]
            if f.filename:
                saved_path = save_upload(f)
                filename = saved_path.name
                img = Image.open(str(saved_path))
                log.debug("Upload saved: %s  size=%s", saved_path, Image.open(str(saved_path)).size)
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            force_online = bool(data.get("force_online"))
            b64 = data.get("image", "")
            if b64:
                img = decode_base64_image(b64)
                saved_path = save_pil(img)
                filename = saved_path.name

        if img is None:
            return jsonify({"ok": False, "error": "No image provided"}), 400

        log.debug("Predict endpoint: calling pipeline with filename=%s  force_online=%s", filename, force_online)
        log.debug("Loaded models summary: CNN=%s  YOLO=%s", getattr(MODELS, 'cnn_model_paths', []), YOLO_PATH)
        return jsonify(_run_full_pipeline(img, filename, source="upload", force_online=force_online))

    except Exception as e:
        log.error("Predict route error: %s\n%s", e, traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Lightweight endpoint for live-camera frames (base64 JSON)."""
    try:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image", "")
        if not b64:
            return jsonify({"ok": False, "error": "no image"}), 400

        img = decode_base64_image(b64)
        log.debug("Camera frame received: %s", img.size)
        # Don't always save live frames; only save if requested
        save_it = bool(data.get("save"))
        filename = ""
        if save_it:
            p = save_pil(img)
            filename = p.name
        return jsonify(_run_full_pipeline(img, filename, source="camera"))

    except Exception as e:
        log.error("predict_frame error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/debug_status", methods=["GET"])
def debug_status():
    """Return loaded model and class info for debugging."""
    try:
        return jsonify({
            "cnn_models": getattr(MODELS, 'cnn_model_paths', []),
            "cnn_count": len(getattr(MODELS, 'cnn_models', [])),
            "yolo_path": str(YOLO_PATH),
            "yolo_names": getattr(MODELS.yolo_model, 'names', {}) if MODELS.yolo_model else {},
            "classes_count": len(CLASSES),
            "classes_sample": CLASSES[:10],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_full_pipeline(img: Image.Image,
                       filename: str,
                       source: str = "upload",
                       force_online: bool = False) -> Dict[str, Any]:
    """Common path: CNN ensemble -> YOLO -> KB lookup -> web fallback.

    The live-camera path is more aggressive about bounding boxes and uses
    additional fallback heuristics to ensure a box is shown for a seed-like
    object when possible.
    """
    t0 = time.time()
    t_standardize = 0
    t_cnn = 0
    t_yolo = 0
    t_classify = 0
    
    # 0) Keep the original upload frame for robust full-frame CNN fallback,
    #    but standardize the image before YOLO/crop processing.
    original_img = img
    img = standardize_frame_input(img, source)
    t_standardize = time.time() - t0
    log.debug("Frame standardized: %s from %s (%.1fms)", img.size, source, t_standardize * 1000)

    # 1) CNN ensemble (camera: full-frame CNN first; upload: defer to crop-based CNN)
    t1 = time.time()
    if source == "camera":
        cls, conf, top5 = MODELS.predict_cnn(img)
        frame_cls, frame_conf = cls, conf
        t_cnn = time.time() - t1
        log.debug("CNN prediction (full-frame camera): %s (conf=%.3f) in %.1fms", cls, conf, t_cnn * 1000)
    else:
        # for uploads we will prefer running CNN on YOLO crops (faster)
        cls, conf, top5 = "", 0.0, []
        frame_cls, frame_conf = "", 0.0
        t_cnn = 0.0

    # 2) YOLO bounding boxes (extra detail – overlay on UI)
    t2 = time.time()
    yolo_conf = CAMERA_YOLO_CONF_THRESHOLD if source == "camera" else UPLOAD_YOLO_CONF_THRESHOLD
    raw_yolo_dets = MODELS.predict_yolo(img, conf=yolo_conf)
    raw_yolo_count = len(raw_yolo_dets)
    t_yolo = time.time() - t2
    log.debug("YOLO detection: %d boxes in %.1fms", raw_yolo_count, t_yolo * 1000)
    # If YOLO misses on uploads, try a simple foreground proposal as a fallback
    if source != "camera" and not raw_yolo_dets:
        try:
            prop = propose_seed_boxes(img)
            if prop:
                log.debug("Upload fallback: propose_seed_boxes returned %d proposals", len(prop))
                raw_yolo_dets = raw_yolo_dets + prop
                raw_yolo_count = len(raw_yolo_dets)
        except Exception as e:
            log.debug("propose_seed_boxes upload fallback failed: %s", e)
        # Try a more aggressive proposal generator for phone images
        if not raw_yolo_dets:
            try:
                prop2 = propose_seed_boxes_aggressive(img)
                if prop2:
                    log.debug("Aggressive proposals returned %d", len(prop2))
                    raw_yolo_dets = raw_yolo_dets + prop2
                    raw_yolo_count = len(raw_yolo_dets)
            except Exception as e:
                log.debug("aggressive proposals failed: %s", e)
    # If CNNs are entirely unavailable, try a more permissive YOLO pass
    if not MODELS.cnn_models and not raw_yolo_dets:
        try:
            log.debug("No CNNs loaded: running permissive YOLO fallback (conf=0.20)")
            alt = apply_nms(MODELS.predict_yolo(img, conf=0.20), 0.45)
            if alt:
                log.debug("Permissive YOLO found %d boxes", len(alt))
                raw_yolo_dets = raw_yolo_dets + alt
                raw_yolo_count = len(raw_yolo_dets)
        except Exception as e:
            log.debug("Permissive YOLO fallback failed: %s", e)
    proposal_dets: List[Dict[str, Any]] = []
    if source == "camera" and not raw_yolo_dets:
        proposal_dets = propose_seed_boxes(img)
        if proposal_dets:
            raw_yolo_dets = raw_yolo_dets + proposal_dets

    used_source = ""
    # Fast accept path for uploads: if YOLO is strong and the class matches a known seed,
    # use the YOLO label immediately. Also accept lower-confidence YOLO predictions when
    # the CNN ensemble is unavailable or no crop-based prediction is present.
    if source != "camera" and raw_yolo_dets:
        best = max(raw_yolo_dets, key=lambda d: float(d.get("conf", 0.0)))
        best_conf = float(best.get("conf", 0.0))
        best_class = best.get("class")
        if _seed_key(best_class) in KNOWN_SEED_KEYS:
            if best_conf >= UPLOAD_YOLO_FAST_ACCEPT:
                cls = best_class
                conf = best_conf
                used_source = "yolo_quick_accept"
                log.debug("Upload fast-accepted YOLO label: %s (conf=%.3f)", cls, conf)
            elif not MODELS.cnn_models and best_conf >= UPLOAD_YOLO_MIN_ACCEPT:
                cls = best_class
                conf = best_conf
                used_source = "yolo_quick_accept_forced"
                log.warning("No CNNs present: forcing YOLO label acceptance at lower threshold: %s (conf=%.3f)", cls, conf)
        else:
            if not MODELS.cnn_models and best_conf >= UPLOAD_YOLO_MIN_ACCEPT:
                cls = best_class
                conf = best_conf
                used_source = "yolo_quick_accept_forced"
                log.warning("No CNNs present: accepted YOLO label despite unknown exact class token: %s (conf=%.3f)", cls, conf)
            else:
                log.debug("Upload fast accept rejected unknown YOLO class: %s", best_class)

    crop_threshold = NO_SEED_THRESH
    t3 = time.time()
    yolo_dets, crop_prediction = classify_yolo_seed_crops(
        img,
        raw_yolo_dets,
        min_class_conf=crop_threshold,
        min_yolo_conf=yolo_conf,
        frame_label=cls if source == "camera" else "",
        frame_conf=conf if source == "camera" else 0.0,
    )
    t_classify = time.time() - t3
    yolo_dets = apply_nms(yolo_dets, 0.40)  # Stricter NMS threshold for trained model
    log.debug("Crop classification: %d detections (nms@0.40) in %.1fms", len(yolo_dets), t_classify * 1000)

    if source == "camera" and yolo_dets:
        width, height = img.size
        img_area = float(width * height)
        valid_dets = []
        
        for d in yolo_dets:
            bbox = d.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = [float(v) for v in bbox]
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            box_area = box_w * box_h
            class_conf = float(d.get("class_conf", 0.0))
            
            # Stricter validation for camera detections with trained model
            # Require: decent size, inside image bounds, good CNN confidence
            # Large full-frame detections are valid when the crop is very confident.
            large_box = box_area > 0.40 * img_area
            if (
                box_w >= 14 and box_h >= 14 and                    # Minimum seed size
                box_area >= 0.0025 * img_area and                  # Minimum relative area
                (not large_box or (box_area <= 1.02 * img_area and class_conf >= max(NO_SEED_THRESH, 0.50))) and
                class_conf >= max(NO_SEED_THRESH, 0.25)            # Good CNN confidence
            ):
                valid_dets.append(d)
                log.debug(
                    "Camera det: %s (cnn=%.2f, yolo=%.2f, area=%.1f%% of frame)%s",
                    d.get("class"), class_conf, float(d.get("conf", 0.0)),
                    box_area / img_area * 100,
                    " [large-box accepted]" if large_box else ""
                )
        
        yolo_dets = valid_dets
        log.debug("Camera detections after validation: %d boxes", len(yolo_dets))
        
        if len(yolo_dets) > MAX_CAMERA_DETECTIONS:
            yolo_dets = yolo_dets[:MAX_CAMERA_DETECTIONS]
        
        yolo_dets = choose_best_camera_detection(yolo_dets, "", 0.0)
        yolo_dets = apply_nms(yolo_dets, 0.35)
        log.debug("Camera detections after best-selection NMS: %d boxes", len(yolo_dets))

    if source == "camera" and not yolo_dets:
        # If seed boxes were not reliably found by YOLO, keep the frame-level
        # classification handy, but do not force it onto the detected boxes.
        yolo_dets = []

        # Strong frame-only fallback: when there is no good crop box but the
        # frame-level CNN is very confident and raw YOLO saw seeds, label the
        # best candidate region with the frame seed name.
        if cls and conf >= CAMERA_FALLBACK_THRESHOLD and raw_yolo_dets:
            fallback_boxes = label_seed_boxes_from_frame_prediction(img, raw_yolo_dets, cls, conf)
            if fallback_boxes:
                yolo_dets = fallback_boxes
                used_source = "camera_frame_fallback"

    crop_used = False
    if crop_prediction:
        crop_cls, crop_conf, crop_top5 = crop_prediction
        if source == "camera":
            crop_conf_threshold = max(0.35, conf - 0.10)
            if crop_conf >= crop_conf_threshold:
                cls, conf, top5 = crop_prediction
                crop_used = True
        else:
            # upload: if raw YOLO proposals are ambiguous or cover nearly the
            # whole image, disambiguate with a strong full-frame prediction.
            if crop_prediction:
                cls_f, conf_f, top5_f = MODELS.predict_cnn(original_img)
                if (
                    conf_f >= max(UPLOAD_CONF_THRESHOLD * 0.9, 0.35) and
                    cls_f != crop_cls and
                    conf_f >= crop_conf + 0.08
                ):
                    disambiguate = False
                    if len(raw_yolo_dets) > 1:
                        large_box = any(
                            (max(0.0, (float(d.get("bbox")[2]) - float(d.get("bbox")[0])) *
                             (float(d.get("bbox")[3]) - float(d.get("bbox")[1])))) /
                            float(img.width * img.height) >= 0.55
                            for d in raw_yolo_dets if d.get("bbox") and len(d.get("bbox")) == 4
                        )
                        disambiguate = crop_conf < 0.90 or large_box
                    elif len(raw_yolo_dets) == 1:
                        bbox = raw_yolo_dets[0].get("bbox", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = [float(v) for v in bbox]
                            area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(img.width * img.height)
                            disambiguate = area_ratio >= 0.55
                    if disambiguate:
                        cls, conf, top5 = cls_f, conf_f, top5_f
                        used_source = "upload_frame_disambiguation"
                        crop_used = False
                        crop_prediction = None
            if crop_prediction:
                # accept a strong crop prediction; if it differs from a fast YOLO
                # label, use a lower acceptance threshold for upload images.
                if cls and crop_prediction and crop_prediction[0] != cls and used_source == "yolo_quick_accept":
                    upload_crop_conf_threshold = max(0.50, min(conf + 0.05, 0.70))
                elif cls and crop_prediction and crop_prediction[0] != cls:
                    upload_crop_conf_threshold = max(0.55, min(conf + 0.10, 0.75))
                else:
                    upload_crop_conf_threshold = max(0.55, min(conf + 0.10, 0.75))
                if crop_conf >= upload_crop_conf_threshold:
                    cls, conf, top5 = crop_prediction
                    crop_used = True

    if source == "camera" and yolo_dets and cls and conf >= CAMERA_CLASS_THRESHOLD:
        # Only apply the frame label to a single box when the box's own CNN score
        # is clearly weaker than the strong frame-level prediction.
        if len(yolo_dets) == 1:
            det = yolo_dets[0]
            box_class_conf = float(det.get("class_conf", 0.0))
            if not crop_used and box_class_conf < max(0.35, conf - 0.10):
                det["class"] = cls
                det["seed_class"] = cls
                det["class_conf"] = max(box_class_conf, float(conf))
                det["conf"] = max(float(det.get("conf", 0.0)), float(conf))
                det["fallback"] = True

    # If upload and crop did not yield a confident prediction, run a single full-frame CNN pass as fallback
    if source != "camera" and not crop_used and (not cls):
        t_cnn_start = time.time()
        cls_f, conf_f, top5_f = MODELS.predict_cnn(original_img)
        t_cnn += time.time() - t_cnn_start
        log.debug("CNN prediction (upload fallback): %s (conf=%.3f) in %.1fms", cls_f, conf_f, (time.time()-t_cnn_start)*1000)
        # Use full-frame result only if confidence is reasonable
        if conf_f >= max(UPLOAD_CONF_THRESHOLD * 0.9, 0.35):
            cls, conf, top5 = cls_f, conf_f, top5_f

    # Only set used_source if it hasn't been determined yet
    if not used_source:
        used_source = "crop_cnn_yolo" if crop_used else "ensemble_cnn"
    web_info: Dict[str, Any] = {}
    no_seed = False
    external_seed = False
    online_researched = False
    online_pending = False
    online_top5: List[Dict[str, Any]] = []
    

    # 3a) NO-SEED GATE
    #     If the CNN has very low max-prob AND YOLO didn't surface any
    #     seed bounding box, then the frame probably doesn't contain a
    #     seed at all (it could be a face, a hand, a wall etc.).
    #     We return an explicit no_seed signal instead of forcing a wrong
    #     label – the frontend uses this to display a friendly message.
    if source == "camera" and not yolo_dets:
        no_seed = True
        cls = ""
        conf = 0.0
        top5 = []
        used_source = "no_seed_in_frame"

    elif source == "camera" and yolo_dets and cls and conf >= CAMERA_CLASS_THRESHOLD:
        # The seed-trained YOLO model supplies the box as a generic "seed";
        # the CNN supplies the specific crop name for the live result.
        if not crop_used:
            used_source = "camera_cnn_yolo"

    # 3b) Confidence gate – try offline DB first, otherwise web fallback if requested
    else:
        threshold = CONF_THRESHOLD if source == "camera" else UPLOAD_CONF_THRESHOLD

        # Determine the primary candidate label to check against offline DB
        candidate = None
        if cls:
            candidate = cls
        elif yolo_dets:
            candidate = yolo_dets[0].get("class")
        elif raw_yolo_dets:
            best_raw = raw_yolo_dets[0]
            best_conf = float(best_raw.get("conf", 0.0))
            if source != "camera":
                if best_conf >= UPLOAD_YOLO_FAST_ACCEPT:
                    candidate = best_raw.get("class")
                elif best_conf >= max(UPLOAD_YOLO_CONF_THRESHOLD, 0.30):
                    # Allow a lower-confidence YOLO-only upload fallback when the CNN
                    # crop path did not produce a prediction. This reduces false
                    # unknown responses for upload photos.
                    candidate = best_raw.get("class")
                    used_source = "yolo_raw_fallback"
            else:
                candidate = best_raw.get("class")

        if candidate:
            if source != "camera" and _seed_key(candidate) not in KNOWN_SEED_KEYS:
                # Reject non-seed YOLO candidates on upload fallback
                candidate = None
                log.debug("Upload raw YOLO fallback candidate rejected as non-seed: %s", candidate)
            else:
                kb_entry = get_seed_info(candidate)
                if kb_entry:
                    cls = candidate
                    used_source = "offline_db_lookup"
                    external_seed = False
                else:
                    external_seed = True
                    # Keep the candidate class name so upload results still show
                    # a predicted label even when the offline DB has no extra info.
                    if not cls:
                        cls = candidate

        if force_online:
            candidates = [(candidate or (cls or "unknown"), conf)] + top5[:5]
            web_info, online_top5 = online_research_seed(img, candidates)
            online_researched = True
            if web_info.get("seed_data_available") and web_info.get("online_confidence", 0) >= ONLINE_CONF_THRESHOLD:
                cls = web_info.get("seed_name") or (candidate or cls)
                used_source = "web_fallback"
            else:
                # Keep the model prediction even if online confidence is low
                used_source = "online_low_confidence"
        else:
            if external_seed:
                used_source = "offline_db_miss"
                # For upload: don't mark as pending if we have a valid prediction
                if source != "upload" or not cls:
                    online_pending = True

    # 4) Karnataka knowledge base
    kb = {} if external_seed else (get_seed_info(cls) if cls else {})

    upload_threshold = CONF_THRESHOLD if source == "camera" else UPLOAD_CONF_THRESHOLD
    # Final fallback: if we still have no class prediction, try YOLO/ proposals
    if not cls:
        # Prefer confident YOLO detections
        if yolo_dets:
            for det in sorted(yolo_dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True):
                lab = det.get('class') or det.get('label') or det.get('name')
                if lab and _seed_key(lab) in KNOWN_SEED_KEYS:
                    cls = lab
                    conf = float(det.get('conf', 0.0))
                    used_source = 'yolo_final_fallback'
                    external_seed = False
                    break
        # If still nothing, try proposals and classify them if CNNs are available
        if not cls and proposal_dets:
            if MODELS.cnn_models:
                for pdet in sorted(proposal_dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True):
                    try:
                        x1, y1, x2, y2 = [int(v) for v in pdet.get('bbox', [])]
                        crop = img.convert('RGB').crop((x1, y1, x2, y2))
                        ccls, cconf, ctop5 = MODELS.predict_cnn(crop)
                        if ccls and cconf >= NO_SEED_THRESH:
                            cls = ccls
                            conf = cconf
                            used_source = 'proposal_cnn_fallback'
                            external_seed = False
                            break
                    except Exception:
                        continue
    if source != "camera" and not no_seed and not yolo_dets and cls and conf >= upload_threshold:
        yolo_dets = approximate_seed_box(img, cls or "seed", conf)

    # 5) build response
    result = {
        "ok": True,
        "no_seed": no_seed,
        "external_seed": external_seed,
        "out_of_database": external_seed,
        "online_pending": online_pending,
        "online_researched": online_researched,
        "seed_data_available": bool(web_info.get("seed_data_available")) if online_researched else False,
        "online_confidence": round(float(web_info.get("online_confidence", 0.0)), 4) if online_researched else 0.0,
        "online_confidence_threshold": ONLINE_CONF_THRESHOLD,
        "out_of_database_message": OFFLINE_DB_MSG if external_seed else "",
        "connect_message": CONNECT_MSG if external_seed else "",
        "seed_not_found_message": SEED_NOT_FOUND_MSG if external_seed else "",
        "low_confidence_message": LOW_ONLINE_CONF_MSG if external_seed else "",
        "predicted_class": cls,
        "confidence": round(float(web_info.get("online_confidence", conf)), 4) if online_researched else round(conf, 4),
        "top5": online_top5 if online_researched else [{"class": c, "confidence": round(p, 4)} for c, p in top5],
        "yolo": yolo_dets,
        "debug_detection": {
            "raw_yolo_count": raw_yolo_count,
            "proposal_count": len(proposal_dets),
            "labeled_box_count": len(yolo_dets),
            "timing_ms": {
                "standardize": round(t_standardize * 1000, 1),
                "cnn": round(t_cnn * 1000, 1),
                "yolo": round(t_yolo * 1000, 1),
                "classify": round(t_classify * 1000, 1),
            },
            "camera_debug": {
                "frame_prediction": {
                    "class": frame_cls,
                    "confidence": round(frame_conf, 4),
                },
                "used_source": used_source,
                "no_seed": no_seed,
                "thresholds": {
                    "yolo_conf_threshold": yolo_conf,
                    "camera_class_threshold": CAMERA_CLASS_THRESHOLD,
                    "camera_fallback_threshold": CAMERA_FALLBACK_THRESHOLD,
                    "no_seed_threshold": NO_SEED_THRESH,
                },
            },
        } if source == "camera" else {},
        "karnataka_info": kb,
        "web_info": web_info,
        "source": used_source,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "filename": filename,
        "image_url": (url_for("static", filename=f"uploads/{filename}")
                      if filename else ""),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    # 6) history – skip when frame has no seed at all
    if not no_seed and cls and not online_pending:
        append_history({
            "id": uuid.uuid4().hex[:10],
            "timestamp": result["timestamp"],
            "filename": filename,
            "predicted_class": cls,
            "confidence": result["confidence"],
            "source": f"{source}/{used_source}",
            "top5": json.dumps(result["top5"]),
            "notes": (kb.get("notes", "") if kb else "")[:120],
        })
    # Debug summary: print final decision and model/class mapping
    try:
        log.info("Final prediction: class=%s  conf=%.4f  source=%s", cls, float(conf or 0.0), used_source)
        log.debug("Models loaded: CNN=%s  YOLO=%s", getattr(MODELS, 'cnn_model_paths', []), YOLO_PATH)
        log.debug("Classes (%d): %s", len(CLASSES), CLASSES[:10])
        log.debug("KNOWN_SEED_KEYS sample: %s", list(KNOWN_SEED_KEYS)[:10])
    except Exception:
        pass
    return result


# ------------------------------------------------------- info endpoints ---
@app.route("/api/seed_info/<name>")
def api_seed_info(name):
    info = get_seed_info(name)
    if not info:
        info = {"error": "not in Karnataka DB", "name": name}
    return jsonify(info)


@app.route("/api/classes")
def api_classes():
    return jsonify({"classes": CLASSES, "count": len(CLASSES)})


@app.route("/api/karnataka_db")
def api_karnataka_db():
    return jsonify(KARNATAKA_DB)


@app.route("/api/history")
def api_history():
    return jsonify({"rows": read_history()[::-1]})


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    clear_history()
    return jsonify({"ok": True})


@app.route("/api/export_history")
def api_export_history():
    if not HISTORY_FILE.exists():
        return Response("No history yet", mimetype="text/plain")
    return send_from_directory(str(BASE_DIR), HISTORY_FILE.name,
                               as_attachment=True)


@app.route("/api/debug")
def api_debug():
    return jsonify({
        "ok": True,
        "yolo_path": str(YOLO_PATH),
        "yolo_loaded": MODELS.yolo_model is not None,
        "yolo_classes": len(MODELS.yolo_model.names) if MODELS.yolo_model else 0,
        "yolo_class_sample": list(MODELS.yolo_model.names.values())[:12] if MODELS.yolo_model else [],
        "cnn_models_loaded": len(MODELS.cnn_models),
        "classes_count": len(CLASSES),
        "classes_sample": CLASSES[:12],
        "seed_only_mode": SEED_ONLY_MODE,
        "known_seed_keys": len(KNOWN_SEED_KEYS),
        "thresholds": {
            "CONF_THRESHOLD": CONF_THRESHOLD,
            "NO_SEED_THRESH": NO_SEED_THRESH,
            "YOLO_CONF_THRESHOLD": YOLO_CONF_THRESHOLD,
            "CAMERA_YOLO_CONF_THRESHOLD": CAMERA_YOLO_CONF_THRESHOLD,
            "CAMERA_CLASS_THRESHOLD": CAMERA_CLASS_THRESHOLD,
            "CAMERA_FALLBACK_THRESHOLD": CAMERA_FALLBACK_THRESHOLD,
        },
        "endpoints": {
            "upload": "/predict",
            "camera_frame": "/predict_frame",
            "health": "/api/health",
        },
        "uptime_started": SERVER_START.isoformat(),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "tensorflow": TF_AVAILABLE,
        "yolo": YOLO_AVAILABLE,
        "web_search": WEB_AVAILABLE,
        "cnn_models_loaded": len(MODELS.cnn_models),
        "yolo_loaded": MODELS.yolo_model is not None,
        "classes": len(CLASSES),
        "uptime_started": SERVER_START.isoformat(),
    })


# --------------------------------------------------- error handlers --------
@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": "File too large (>16MB)"}), 413


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "not found"}), 404
    return render_template("index.html",
                           classes=CLASSES,
                           n_classes=len(CLASSES),
                           karnataka_db=KARNATAKA_DB), 404


@app.errorhandler(500)
def internal_err(e):
    log.error("500 ERROR: %s\n%s", e, traceback.format_exc())
    return jsonify({"ok": False, "error": "internal server error"}), 500


# ============================================================================
#                                MAIN
# ============================================================================

SERVER_START = datetime.datetime.utcnow()


def _find_chrome_executable() -> Optional[Path]:
    """Locate Chrome on Windows, falling back to PATH if available."""
    chrome_from_path = shutil.which("chrome") or shutil.which("chrome.exe")
    if chrome_from_path:
        return Path(chrome_from_path)

    candidates = [
        os.environ.get("CHROME_PATH"),
        str(Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def open_browser_when_ready(url: str, timeout: int = 25) -> None:
    """Wait for Flask to accept requests, then open the app in Chrome."""
    def _worker() -> None:
        health_url = f"{url.rstrip('/')}/api/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=1):
                    break
            except Exception:
                time.sleep(0.5)

        try:
            chrome_path = _find_chrome_executable()
            if chrome_path:
                webbrowser.register(
                    "chrome",
                    None,
                    webbrowser.BackgroundBrowser(str(chrome_path)),
                )
                webbrowser.get("chrome").open_new_tab(url)
            else:
                opened = webbrowser.open_new_tab(url)
                if not opened and os.name == "nt":
                    os.startfile(url)
        except Exception as e:
            log.warning("Could not open browser automatically: %s", e)
            try:
                if os.name == "nt":
                    os.startfile(url)
                else:
                    webbrowser.open_new_tab(url)
            except Exception as inner:
                log.warning("Fallback browser open failed: %s", inner)
                print(f"Open this URL in your browser: {url}")
        else:
            print(f"If the browser does not open automatically, visit: {url}")

    threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    local_url = "http://127.0.0.1:5000"
    log.info("Starting Flask dev server at %s", local_url)
    log.info("CNN ensemble : %d model(s)", len(MODELS.cnn_models))
    log.info("YOLO         : %s", "loaded" if MODELS.yolo_model else "—")
    log.info("Classes      : %d", len(CLASSES))
    print("\n" + "=" * 70)
    print(" REAL-TIME AGRICULTURE SEEDS DETECTION & RECOGNITION USING AI")
    print(" Flask backend ready  →  http://127.0.0.1:5000")
    print(" CNN models : {}  |  YOLO : {}  |  Classes : {}".format(
        len(MODELS.cnn_models),
        "yes" if MODELS.yolo_model else "no",
        len(CLASSES)))
    print("=" * 70 + "\n")
    open_browser_when_ready(local_url)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
