"""Generate a toy synthetic dataset for pipeline smoke-testing only.

These geometric drawings are NOT a substitute for real civic-issue photographs.
Do not use a model trained only on this dataset for real-world decisions.
"""
from pathlib import Path
import random

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "dataset" / "synthetic_images"
LABEL_DIR = BASE_DIR / "dataset" / "synthetic_labels"
CATEGORIES = ["pothole", "construction", "garbage", "wires", "streetlight", "good_place"]
SAMPLES_PER_CLASS = 300
IMAGE_SIZE = 128
SEED = 42


def make_image(category, rng):
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 150, dtype=np.uint8)
    # Small random variation prevents every image in a class being identical.
    image = np.clip(image.astype(np.int16) + rng.integers(-12, 13, image.shape), 0, 255).astype(np.uint8)

    if category == "pothole":
        cv2.ellipse(image, (rng.integers(42, 86), rng.integers(42, 86)),
                    (rng.integers(18, 38), rng.integers(12, 30)),
                    rng.integers(0, 180), 0, 360, (25, 25, 25), -1)
    elif category == "construction":
        for _ in range(3):
            x, y = int(rng.integers(8, 70)), int(rng.integers(8, 65))
            cv2.rectangle(image, (x, y), (min(x + 35, 127), min(y + 48, 127)), (105, 95, 75), -1)
            cv2.line(image, (x, y), (min(x + 35, 127), y), (220, 180, 80), 2)
    elif category == "garbage":
        for _ in range(18):
            x, y = int(rng.integers(8, 120)), int(rng.integers(8, 120))
            color = tuple(int(v) for v in rng.integers(20, 240, 3))
            cv2.circle(image, (x, y), int(rng.integers(3, 9)), color, -1)
    elif category == "wires":
        for _ in range(3):
            y1, y2 = int(rng.integers(15, 110)), int(rng.integers(15, 110))
            cv2.line(image, (0, y1), (127, y2), (15, 15, 15), int(rng.integers(1, 4)))
    elif category == "streetlight":
        x = int(rng.integers(45, 84))
        cv2.line(image, (x, 42), (x, 124), (45, 45, 45), 4)
        cv2.circle(image, (x, 35), 12, (255, 230, 30), -1)
        cv2.line(image, (x, 35), (x + 22, 35), (45, 45, 45), 3)
    else:  # good_place
        cv2.rectangle(image, (12, 12), (116, 116), (70, 130, 70), 3)
        cv2.circle(image, (64, 64), 25, (30, 210, 70), -1)
        cv2.line(image, (35, 90), (95, 90), (230, 230, 230), 3)

    return image


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # Remove previous generated files so stale samples cannot be loaded accidentally.
    for folder in (IMAGE_DIR, LABEL_DIR):
        for path in folder.glob("sample_*.jpg" if folder == IMAGE_DIR else "sample_*.txt"):
            path.unlink()

    index = 0
    for class_index, category in enumerate(CATEGORIES):
        for _ in range(SAMPLES_PER_CLASS):
            image = make_image(category, rng)
            image_path = IMAGE_DIR / f"sample_{index:05d}.jpg"
            label_path = LABEL_DIR / f"sample_{index:05d}.txt"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Could not write {image_path}")
            label_path.write_text(str(class_index), encoding="utf-8")
            index += 1

    print(f"Generated {index} synthetic images across {len(CATEGORIES)} classes.")
    print("WARNING: This is only a pipeline test dataset, not real-world training data.")


if __name__ == "__main__":
    main()
