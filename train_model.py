"""Train NagarNetra with MobileNetV2 transfer learning.

Required dataset structure:
dataset/real/
  potholes/
  illegal_buildings/
  sanitation/
  garbage/
  street_lights/
  hanging_wires/

Do not use the synthetic images from generate_data.py for real-world training.
"""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "real"
MODEL_PATH = BASE_DIR / "model_v5.keras"
METADATA_PATH = BASE_DIR / "model_metadata.json"
IMAGE_SIZE = (224, 224)
CATEGORIES = ["potholes", "illegal_buildings", "sanitation", "garbage", "street_lights", "hanging_wires"]
SEED = 42
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_gpu():
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"GPU memory growth enabled: {device.name}")
        except RuntimeError as exc:
            print(f"GPU configuration note: {exc}")


def collect_real_images():
    paths, labels, counts = [], [], {}
    missing = []
    for class_index, category in enumerate(CATEGORIES):
        folder = DATA_DIR / category
        if not folder.exists():
            missing.append(category)
            counts[category] = 0
            continue
        class_files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[category] = len(class_files)
        paths.extend(str(p) for p in class_files)
        labels.extend([class_index] * len(class_files))

    print("Real-image dataset counts:")
    for category, count in counts.items():
        print(f"  {category:14s}: {count}")

    if missing:
        raise RuntimeError(
            "Missing class folders under dataset/real: " + ", ".join(missing)
        )
    too_small = [name for name, count in counts.items() if count < 10]
    if too_small:
        raise RuntimeError(
            "At least 10 real images per class are required for a stratified "
            "train/validation/test split. Add images for: " + ", ".join(too_small)
        )
    return np.asarray(paths), np.asarray(labels, dtype=np.int32), counts


def decode_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    # Keep pixels in [0,255]. Rescaling to [-1,1] is inside the saved model.
    return tf.cast(image, tf.float32), label


def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_rgb_0_255")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.06)(x)
    x = tf.keras.layers.RandomZoom(0.12)(x)
    # MobileNetV2 expects [-1,1]. Keep preprocessing embedded for inference parity.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocessing")(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(len(CATEGORIES), activation="softmax", name="category_output")(x)
    model = tf.keras.Model(inputs, outputs, name="NagarNetra_MobileNetV2")
    return model, backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    args = parser.parse_args()

    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    paths, labels, counts = collect_real_images()

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.1765, random_state=SEED, stratify=train_labels
    )

    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels, training=False)
    test_ds = make_dataset(test_paths, test_labels, training=False)

    model, backbone = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
    ]

    print(f"Training={len(train_paths)}, validation={len(val_paths)}, test={len(test_paths)}")
    print("Stage 1: train classification head")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs,
        class_weight=class_weights, callbacks=callbacks, verbose=1
    )

    if args.fine_tune_epochs > 0:
        print("Stage 2: fine-tune upper MobileNetV2 layers")
        backbone.trainable = True
        for layer in backbone.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        # Keep BatchNorm frozen for more stable small-dataset fine-tuning.
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        fine_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
        ]
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            class_weight=class_weights, callbacks=fine_callbacks, verbose=1
        )

    best_model = tf.keras.models.load_model(MODEL_PATH) if MODEL_PATH.exists() else model
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    print(f"Real-photo holdout test loss: {test_loss:.4f}")
    print(f"Real-photo holdout test accuracy: {test_accuracy:.4f}")
    best_model.save(MODEL_PATH)

    metadata = {
        "architecture": "MobileNetV2 transfer learning",
        "image_size": list(IMAGE_SIZE),
        "categories": CATEGORIES,
        "input_format": "RGB float32 pixels in range 0..255; preprocessing embedded in model",
        "training_data": "dataset/real",
        "class_counts": counts,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "confidence_threshold": 0.60,
        "synthetic_data_used": False
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print("Important: inspect per-class precision/recall and test on independent local images before deployment.")


if __name__ == "__main__":
    main()
