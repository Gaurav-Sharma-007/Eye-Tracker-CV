"""
train_model.py
──────────────
Train a MobileNetV2-based gaze direction classifier on the dataset in data/.

5 classes: up | down | straight | left | right

Pipeline
─────────
1. Scan data/<class>/ folders for all .jpg images
2. Stratified train/val/test split  (70 / 15 / 15)
3. tf.data pipeline with augmentation (train only)
4. MobileNetV2 backbone (ImageNet weights, frozen initially)
5. Two-phase training:
   Phase 1 – train head only   (5  epochs)
   Phase 2 – fine-tune top 40 layers (30 epochs, lower lr)
6. EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
7. Test-set evaluation + classification report
8. Save final model → models/gaze_model.keras
9. Save class-index mapping → models/class_indices.json
10. Optionally use Amazon Rekognition for high-accuracy face cropping
    before inference (set USE_REKOGNITION=True below)

Run:
    python train_model.py
"""

import os
import json
import random
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DATA_DIR        = pathlib.Path("data")
MODEL_DIR       = pathlib.Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Image size for MobileNetV2
IMG_H, IMG_W    = 224, 224

# ── Split ratios ──────────────────────────────────────────────
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15          # remainder

# ── Hyperparameters ───────────────────────────────────────────
BATCH_SIZE      = 32
PHASE1_EPOCHS   = 5             # head-only warmup
PHASE2_EPOCHS   = 40            # fine-tuning (with early-stop)
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-4
UNFREEZE_LAYERS = 40            # top N layers of backbone to unfreeze

# ── Augmentation strength ─────────────────────────────────────
AUG_ROTATION    = 0.10          # ±10 %
AUG_ZOOM        = 0.10
AUG_BRIGHTNESS  = 0.15

# ── Amazon Rekognition face-crop toggle ───────────────────────
# Requires AWS credentials configured (aws configure) and
# the boto3 package installed.  Set to True for inference only;
# training still uses the original images.
USE_REKOGNITION = False         # flip to True when AWS is ready

SEED            = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────
# 1. Discover images
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"[data] Classes found: {CLASS_NAMES}")

all_paths, all_labels = [], []
for idx, cls in enumerate(CLASS_NAMES):
    images = list((DATA_DIR / cls).glob("*.jpg")) + \
             list((DATA_DIR / cls).glob("*.jpeg")) + \
             list((DATA_DIR / cls).glob("*.png"))
    print(f"  {cls:10s}: {len(images)} images  (label={idx})")
    all_paths.extend([str(p) for p in images])
    all_labels.extend([idx] * len(images))

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
print(f"[data] Total images: {len(all_paths)}")

# ─────────────────────────────────────────────────────────────
# 2. Stratified train / val / test split
# ─────────────────────────────────────────────────────────────

train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
    all_paths, all_labels,
    test_size=(1 - TRAIN_RATIO),
    stratify=all_labels,
    random_state=SEED,
)

# Split tmp into val and test (equal halves of 30 %)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    tmp_paths, tmp_labels,
    test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
    stratify=tmp_labels,
    random_state=SEED,
)

print(f"\n[split] Train : {len(train_paths)}")
print(f"[split] Val   : {len(val_paths)}")
print(f"[split] Test  : {len(test_paths)}")

# ─────────────────────────────────────────────────────────────
# 3. tf.data pipeline
# ─────────────────────────────────────────────────────────────

def preprocess(path, label):
    """Load → decode → resize → normalise to [-1, 1] (MobileNetV2)."""
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_jpeg(raw, channels=3)
    img   = tf.image.resize(img, [IMG_H, IMG_W])
    img   = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

def augment(img, label):
    """Light augmentation – applied only during training."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    # Random rotation via keras layer (wrapped in py_function for compat)
    img = tf.keras.layers.RandomRotation(AUG_ROTATION, seed=SEED)(
        tf.expand_dims(img, 0))[0]
    img = tf.keras.layers.RandomZoom(AUG_ZOOM, seed=SEED)(
        tf.expand_dims(img, 0))[0]
    return img, label

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment,    num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds   = make_dataset(val_paths,   val_labels)
test_ds  = make_dataset(test_paths,  test_labels)

n_classes = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────────────
# 4. Model definition
# ─────────────────────────────────────────────────────────────

def build_model(n_classes: int, backbone_trainable=False):
    """MobileNetV2 backbone + custom head."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = backbone_trainable

    inputs = keras.Input(shape=(IMG_H, IMG_W, 3))
    x = backbone(inputs, training=False)          # BN layers frozen in phase 1
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), backbone

model, backbone = build_model(n_classes, backbone_trainable=False)
model.summary(line_length=90)

# ─────────────────────────────────────────────────────────────
# 5. Phase 1 – train head only
# ─────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("Phase 1: Training head (backbone frozen)")
print(f"{'═'*60}")

model.compile(
    optimizer=keras.optimizers.Adam(PHASE1_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_p1 = [
    keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "best_phase1.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1,
    ),
]

hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=callbacks_p1,
)

# ─────────────────────────────────────────────────────────────
# 6. Phase 2 – fine-tune top N backbone layers
# ─────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"Phase 2: Fine-tuning top {UNFREEZE_LAYERS} backbone layers")
print(f"{'═'*60}")

backbone.trainable = True
for layer in backbone.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"  Trainable layers: {trainable_count}")

model.compile(
    optimizer=keras.optimizers.Adam(PHASE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "best_gaze_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=4, verbose=1,
    ),
]

hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=callbacks_p2,
)

# ─────────────────────────────────────────────────────────────
# 7. Test evaluation
# ─────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("Test-set evaluation")
print(f"{'═'*60}")

loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy : {acc*100:.2f}%")
print(f"Test loss     : {loss:.4f}")

# Predictions for classification report
y_true, y_pred = [], []
for imgs, lbls in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_true.extend(lbls.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# ─────────────────────────────────────────────────────────────
# 8. Save final model + class mapping
# ─────────────────────────────────────────────────────────────

final_path = MODEL_DIR / "gaze_model.keras"
model.save(final_path)
print(f"\n[save] Model saved → {final_path}")

class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
idx_path = MODEL_DIR / "class_indices.json"
with open(idx_path, "w") as f:
    json.dump({"class_to_idx": class_indices,
               "idx_to_class": {str(i): n for i, n in enumerate(CLASS_NAMES)}}, f, indent=2)
print(f"[save] Class indices → {idx_path}")

# ─────────────────────────────────────────────────────────────
# 9. Plot training curves
# ─────────────────────────────────────────────────────────────

def plot_history(h1, h2):
    acc1   = h1.history["accuracy"]
    vacc1  = h1.history["val_accuracy"]
    acc2   = h2.history["accuracy"]
    vacc2  = h2.history["val_accuracy"]
    loss1  = h1.history["loss"]
    vloss1 = h1.history["val_loss"]
    loss2  = h2.history["loss"]
    vloss2 = h2.history["val_loss"]

    e1 = range(1, len(acc1) + 1)
    e2 = range(len(acc1) + 1, len(acc1) + len(acc2) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(e1, acc1,  "b-",  label="Train P1")
    axes[0].plot(e1, vacc1, "b--", label="Val P1")
    axes[0].plot(e2, acc2,  "r-",  label="Train P2")
    axes[0].plot(e2, vacc2, "r--", label="Val P2")
    axes[0].axvline(len(acc1), color="gray", linestyle=":", label="Phase boundary")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(e1, loss1,  "b-",  label="Train P1")
    axes[1].plot(e1, vloss1, "b--", label="Val P1")
    axes[1].plot(e2, loss2,  "r-",  label="Train P2")
    axes[1].plot(e2, vloss2, "r--", label="Val P2")
    axes[1].axvline(len(acc1), color="gray", linestyle=":")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = MODEL_DIR / "training_curves.png"
    plt.savefig(out, dpi=120)
    print(f"[plot] Training curves → {out}")

plot_history(hist1, hist2)
print("\n[done] Training complete.")
