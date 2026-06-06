import cv2
import numpy as np
import os
import csv
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage import img_as_float

LBP_RADIUS   = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD   = "uniform"

GABOR_FREQUENCIES  = [0.1, 0.2, 0.3]
GABOR_ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.jfif')


def extract_lbp_features(gray_img):
    lbp = local_binary_pattern(gray_img, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_gabor_features(gray_img):
    features = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            real, _ = gabor(gray_img, frequency=freq, theta=theta)
            features.append(real.mean())
            features.append(real.var())
    return np.array(features)


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = img_as_float(rgb2gray(img_rgb))
    lbp_feat   = extract_lbp_features(gray)
    gabor_feat = extract_gabor_features(gray)
    return np.concatenate([lbp_feat, gabor_feat])


def process_dataset(hair_dir="data/hair", output_csv="data/hair_features.csv"):
    labels = ["3C", "4A", "4B", "4C"]
    rows = []
    skipped = 0
    processed = 0
    feature_names = None

    for label in labels:
        folder = os.path.join(hair_dir, label)
        if not os.path.exists(folder):
            print(f"  ⚠ Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
        print(f"  Processing {label}/ — {len(files)} images...")

        for fname in files:
            features = extract_features(os.path.join(folder, fname))
            if features is None:
                skipped += 1
                continue
            rows.append(list(features) + [label])
            processed += 1
            if feature_names is None:
                n = len(features)
                feature_names = (
                    [f"lbp_{i}"   for i in range(26)] +
                    [f"gabor_{i}" for i in range(n - 26)]
                )

    if not rows:
        print("No images processed. Check folder names.")
        return 0, 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_names + ["label"])
        writer.writerows(rows)

    print(f"\n✓ Done. {processed} processed, {skipped} skipped.")
    print(f"✓ Feature vector size: {len(feature_names)} dimensions")
    print(f"✓ Saved to {output_csv}")
    return processed, skipped


if __name__ == "__main__":
    print("Extracting hair texture features...\n")
    process_dataset()