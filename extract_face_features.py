import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# --- New MediaPipe API (0.10.30+) ---
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request

# Download the face landmarker model if not already present
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe face landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded.")

# --- Setup ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# ── Key landmark indices ────────────────────────────────────────────────────────
FOREHEAD_TOP   = 10
CHIN_BOTTOM    = 152
LEFT_CHEEK     = 234   # widest face point (left)
RIGHT_CHEEK    = 454   # widest face point (right)
LEFT_JAW       = 172   # lower jaw (left)
RIGHT_JAW      = 397   # lower jaw (right)
LEFT_FOREHEAD  = 70    # forehead width (left)
RIGHT_FOREHEAD = 300   # forehead width (right)
LEFT_TEMPLE    = 162
RIGHT_TEMPLE   = 389
# Extra landmarks for richer geometry
LEFT_MIDJAW    = 135   # mid jaw / gonial area (left)
RIGHT_MIDJAW   = 364   # mid jaw / gonial area (right)
CHIN_LEFT      = 149   # chin contour left
CHIN_RIGHT     = 378   # chin contour right
NOSE_BRIDGE    = 168   # between eyes (vertical reference)
UPPER_LIP      = 0     # philtrum top
LEFT_CHEEKBONE = 116   # cheekbone (left)
RIGHT_CHEEKBONE= 345   # cheekbone (right)


def _angle(a, b, c):
    """Angle at point b formed by a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    # Core measurements
    face_height    = np.linalg.norm(pt(FOREHEAD_TOP) - pt(CHIN_BOTTOM))
    face_width     = np.linalg.norm(pt(LEFT_CHEEK) - pt(RIGHT_CHEEK))
    jaw_width      = np.linalg.norm(pt(LEFT_JAW) - pt(RIGHT_JAW))
    forehead_width = np.linalg.norm(pt(LEFT_FOREHEAD) - pt(RIGHT_FOREHEAD))
    temple_width   = np.linalg.norm(pt(LEFT_TEMPLE) - pt(RIGHT_TEMPLE))
    cheekbone_w    = np.linalg.norm(pt(LEFT_CHEEKBONE) - pt(RIGHT_CHEEKBONE))
    midjaw_width   = np.linalg.norm(pt(LEFT_MIDJAW) - pt(RIGHT_MIDJAW))
    chin_width     = np.linalg.norm(pt(CHIN_LEFT) - pt(CHIN_RIGHT))

    if face_width == 0 or face_height == 0 or forehead_width == 0:
        return None

    # Vertical segments (proportions along the face)
    upper_third = np.linalg.norm(pt(FOREHEAD_TOP) - pt(NOSE_BRIDGE))
    mid_third   = np.linalg.norm(pt(NOSE_BRIDGE) - pt(UPPER_LIP))
    lower_third = np.linalg.norm(pt(UPPER_LIP) - pt(CHIN_BOTTOM))

    # Jaw angle (how sharp/round the jaw corner is) — left and right, averaged
    jaw_angle_l = _angle(pt(LEFT_MIDJAW), pt(LEFT_JAW), pt(CHIN_BOTTOM))
    jaw_angle_r = _angle(pt(RIGHT_MIDJAW), pt(RIGHT_JAW), pt(CHIN_BOTTOM))
    jaw_angle   = (jaw_angle_l + jaw_angle_r) / 2.0

    # Chin sharpness (angle at the chin point)
    chin_angle = _angle(pt(CHIN_LEFT), pt(CHIN_BOTTOM), pt(CHIN_RIGHT))

    features = [
        face_height / face_width,        # 1  overall aspect
        jaw_width / face_width,          # 2
        forehead_width / face_width,     # 3
        temple_width / face_width,       # 4
        cheekbone_w / face_width,        # 5
        midjaw_width / face_width,       # 6
        chin_width / face_width,         # 7
        jaw_width / forehead_width,      # 8  taper top->bottom
        jaw_width / cheekbone_w,         # 9
        forehead_width / cheekbone_w,    # 10
        chin_width / jaw_width,          # 11 how pointed the chin is
        jaw_width / face_height,         # 12
        cheekbone_w / face_height,       # 13
        upper_third / face_height,       # 14 vertical proportions
        mid_third / face_height,         # 15
        lower_third / face_height,       # 16
        upper_third / lower_third,       # 17
        jaw_angle / 180.0,               # 18 normalized jaw sharpness
        chin_angle / 180.0,              # 19 normalized chin sharpness
        midjaw_width / jaw_width,        # 20 jaw fullness
    ]
    return features


FEATURE_NAMES = [
    "ratio_hw", "ratio_jaw", "ratio_forehead", "ratio_temple",
    "ratio_cheekbone", "ratio_midjaw", "ratio_chin",
    "jaw_over_forehead", "jaw_over_cheek", "forehead_over_cheek",
    "chin_over_jaw", "jaw_over_height", "cheek_over_height",
    "upper_third", "mid_third", "lower_third", "upper_over_lower",
    "jaw_angle", "chin_angle", "midjaw_over_jaw",
]


def process_dataset(faces_dir="data/faces", output_csv="data/face_features.csv"):
    labels = ["oval", "round", "square", "heart", "oblong"]
    rows = []
    skipped = 0
    processed = 0

    for label in labels:
        folder = os.path.join(faces_dir, label)
        if not os.path.exists(folder):
            print(f"  Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif'))]
        print(f"  Processing {label}/ - {len(files)} images...")
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                features = extract_features(path)
            except Exception:
                features = None
            if features is None:
                skipped += 1
                continue
            rows.append(features + [label])
            processed += 1

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        writer.writerows(rows)

    print(f"\nDone. {processed} processed, {skipped} skipped.")
    print(f"Saved to {output_csv}  ({len(FEATURE_NAMES)} features per face)")


if __name__ == "__main__":
    print("Extracting facial landmark features (expanded set)...\n")
    process_dataset()