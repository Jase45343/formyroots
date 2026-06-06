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
    print("✓ Model downloaded.")

# --- Setup ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Key landmark indices (same as before) ---
FOREHEAD_TOP   = 10
CHIN_BOTTOM    = 152
LEFT_CHEEK     = 234
RIGHT_CHEEK    = 454
LEFT_JAW       = 172
RIGHT_JAW      = 397
LEFT_FOREHEAD  = 70
RIGHT_FOREHEAD = 300
LEFT_TEMPLE    = 162
RIGHT_TEMPLE   = 389

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # New API uses mp.Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    face_height    = np.linalg.norm(pt(FOREHEAD_TOP) - pt(CHIN_BOTTOM))
    face_width     = np.linalg.norm(pt(LEFT_CHEEK) - pt(RIGHT_CHEEK))
    jaw_width      = np.linalg.norm(pt(LEFT_JAW) - pt(RIGHT_JAW))
    forehead_width = np.linalg.norm(pt(LEFT_FOREHEAD) - pt(RIGHT_FOREHEAD))
    temple_width   = np.linalg.norm(pt(LEFT_TEMPLE) - pt(RIGHT_TEMPLE))

    if face_width == 0 or face_height == 0:
        return None

    return [
        face_height / face_width,
        jaw_width / face_width,
        forehead_width / face_width,
        temple_width / face_width,
        jaw_width / forehead_width,
        jaw_width / face_height
    ]


def process_dataset(faces_dir="data/faces", output_csv="data/face_features.csv"):
    labels = ["oval", "round", "square", "heart", "oblong"]
    rows = []
    skipped = 0
    processed = 0

    for label in labels:
        folder = os.path.join(faces_dir, label)
        if not os.path.exists(folder):
            print(f"  ⚠ Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif'))]

        print(f"  Processing {label}/ — {len(files)} images...")

        for fname in files:
            path = os.path.join(folder, fname)
            features = extract_features(path)
            if features is None:
                skipped += 1
                continue
            rows.append(features + [label])
            processed += 1

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ratio_hw", "ratio_jaw", "ratio_forehead",
            "ratio_temple", "ratio_jaw_fh", "ratio_jaw_h", "label"
        ])
        writer.writerows(rows)

    print(f"\n✓ Done. {processed} processed, {skipped} skipped.")
    print(f"✓ Saved to {output_csv}")


if __name__ == "__main__":
    print("Extracting facial landmark features...\n")
    process_dataset()