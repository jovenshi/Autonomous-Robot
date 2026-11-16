"""Train the sign classifier on sorted crops. Laptop only (needs scikit-learn).

Run from the autonomous/ folder:
    python -m tools.train_signs signs_data --holdout signs_left_3,signs_dim_1

signs_data/ holds one folder per class: pause/, left/, right/, straight/,
none/ (crops from tools/collect_signs.py, sorted by hand). Each crop is
augmented SIGN_AUG_COPIES times (brightness, contrast, rotation, scale,
shift, blur; never flipped: a flipped LEFT is a RIGHT), normalised with the
robot's own normalise_crop, described by HOG and fitted with a linear SVM
(LinearSVC). Whole recordings named by --holdout are kept out of training
(neighbouring frames are near-copies); the tool prints their accuracy and
confusion matrix, suggests SIGN_MARGIN_MIN, and writes SIGN_MODEL_PATH.
Copy that file to the Pi.
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

# Let `python tools/<file>.py` find modules in autonomous/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from perception.signs import classify_sign, hog_features, make_hog, normalise_crop
from robot_types import SignModel, SignType

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # autonomous/
CLASSES = tuple(t.name for t in SignType)                # folder = class name in lower case
_FILE = re.compile(r"^(?P<rec>.+)_\d{6}(_[cr]\d+)?\.png$")


# ---------------------------------------------------------------- data

def recording_of(filename):
    # "signs_left_1_000123_c0.png" -> "signs_left_1"; None if not named by collect_signs.
    match = _FILE.match(filename)
    return match.group("rec") if match else None


def load_samples(data_dir):
    # [(crop BGR, class name, recording)] from the class folders that exist.
    samples = []
    for name in CLASSES:
        folder = os.path.join(data_dir, name.lower())
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            crop = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
            if crop is not None:
                samples.append((crop, name, recording_of(filename)))
    return samples


def split_by_recording(samples, holdout):
    # (train, test): test is every sample from a recording named in holdout.
    test = [s for s in samples if s[2] in holdout]
    train = [s for s in samples if s[2] not in holdout]
    return train, test


# ---------------------------------------------------------------- augmentation

def augment(crop, rng, cfg):
    # One randomly changed copy of a BGR crop. Never flipped.
    h, w = crop.shape[:2]
    angle = rng.uniform(-cfg.SIGN_AUG_ROT_DEG, cfg.SIGN_AUG_ROT_DEG)
    scale = 1.0 + rng.uniform(-cfg.SIGN_AUG_SCALE, cfg.SIGN_AUG_SCALE)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += rng.uniform(-cfg.SIGN_AUG_SHIFT_FRAC, cfg.SIGN_AUG_SHIFT_FRAC, 2) * (w, h)
    out = cv2.warpAffine(crop, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    contrast = 1.0 + rng.uniform(-cfg.SIGN_AUG_CONTRAST, cfg.SIGN_AUG_CONTRAST)
    bright = rng.uniform(-cfg.SIGN_AUG_BRIGHT, cfg.SIGN_AUG_BRIGHT)
    out = cv2.convertScaleAbs(out, alpha=contrast, beta=bright)
    blur = int(rng.integers(0, cfg.SIGN_AUG_BLUR_MAX_PX // 2 + 1)) * 2 + 1
    return cv2.GaussianBlur(out, (blur, blur), 0) if blur > 1 else out


def with_augmented(samples, rng, cfg):
    # The originals plus SIGN_AUG_COPIES augmented copies of each.
    out = list(samples)
    for crop, name, rec in samples:
        out.extend((augment(crop, rng, cfg), name, rec) for _ in range(cfg.SIGN_AUG_COPIES))
    return out


# ---------------------------------------------------------------- features and model

def make_training_hog(cfg):
    return make_hog(cfg.SIGN_CROP_PX, cfg.SIGN_HOG_BLOCK_PX, cfg.SIGN_HOG_STRIDE_PX,
                    cfg.SIGN_HOG_CELL_PX, cfg.SIGN_HOG_BINS)


def feature_matrix(samples, hog, cfg):
    # (X, y): one HOG row per crop, through the robot's normalise_crop.
    x = np.array([hog_features(normalise_crop(crop, cfg), hog) for crop, _, _ in samples])
    return x, np.array([name for _, name, _ in samples])


def fit_svm(x, y, cfg):
    # LinearSVC, one-vs-rest -> (weights, biases, classes). scikit-learn is
    # imported here only, so everything else runs on the Pi.
    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=cfg.SIGN_SVM_C, dual=True, max_iter=20000).fit(x, y)
    weights, biases = clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)
    if len(clf.classes_) == 2:              # binary: one row for classes_[1]
        weights, biases = np.vstack([-weights, weights]), np.concatenate([-biases, biases])
    return weights, biases, tuple(str(c) for c in clf.classes_)


def save_model(path, weights, biases, classes, cfg):
    # The file load_sign_model reads: weights, biases, classes, crop size, HOG settings.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, weights=weights, biases=biases, classes=np.array(classes),
             crop_px=cfg.SIGN_CROP_PX,
             hog=np.array([cfg.SIGN_HOG_BLOCK_PX, cfg.SIGN_HOG_STRIDE_PX,
                           cfg.SIGN_HOG_CELL_PX, cfg.SIGN_HOG_BINS]))


# ---------------------------------------------------------------- evaluation

def predict(x, model):
    # [(SignType name, margin)] per row, with classify_sign as on the robot.
    out = []
    for row in x:
        sign, margin, _ = classify_sign(row, model)
        out.append((sign.name, margin))
    return out


def apply_margin(predictions, margin_min):
    # Predictions with the robot's rule: below margin_min -> NONE.
    return [name if margin >= margin_min else "NONE" for name, margin in predictions]


def suggest_margin(predictions, truth):
    # The margin threshold (from those seen) giving the best held-out
    # accuracy; the smallest such one, so as few real signs as possible are lost.
    candidates = sorted({0.0} | {m for _, m in predictions})
    best_t, best_acc = 0.0, -1.0
    for t in candidates:
        acc = float(np.mean([p == g for p, g in zip(apply_margin(predictions, t), truth)]))
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def confusion(predicted, truth, classes):
    # rows = true class, columns = predicted, in classes order.
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), int)
    for p, g in zip(predicted, truth):
        matrix[index[g], index[p]] += 1
    return matrix


def format_confusion(matrix, classes):
    head = "true \\ pred " + " ".join(f"{c[:8]:>8}" for c in classes)
    rows = [f"{c[:11]:<11} " + " ".join(f"{n:8d}" for n in row) for c, row in zip(classes, matrix)]
    return "\n".join([head] + rows)


# ---------------------------------------------------------------- main

def report(model, test, hog, cfg):
    # Held-out accuracy at SIGN_MARGIN_MIN, the confusion matrix and a
    # suggested margin; returns the suggested margin, or None with no test set.
    if not test:
        print("No held-out recordings: name some with --holdout to check the model.")
        return None
    x, y = feature_matrix(test, hog, cfg)
    predictions = predict(x, model)
    predicted = apply_margin(predictions, cfg.SIGN_MARGIN_MIN)
    classes = [c for c in CLASSES if c in set(y) | set(predicted)]
    print(f"Held out: {len(test)} crops, accuracy {np.mean(predicted == y):.3f} "
          f"at SIGN_MARGIN_MIN = {cfg.SIGN_MARGIN_MIN}")
    print(format_confusion(confusion(predicted, y, classes), classes))
    margin, acc = suggest_margin(predictions, list(y))
    print(f"Suggested SIGN_MARGIN_MIN = {margin:.3f} (held-out accuracy {acc:.3f})")
    return margin


def train(data_dir, holdout, out_path, cfg, seed=0):
    samples = load_samples(data_dir)
    train_set, test = split_by_recording(samples, set(holdout))
    if len({name for _, name, _ in train_set}) < 2:
        raise SystemExit(f"Need crops of at least two classes in {data_dir} (outside --holdout)")
    hog = make_training_hog(cfg)
    x, y = feature_matrix(with_augmented(train_set, np.random.default_rng(seed), cfg), hog, cfg)
    print(f"Training on {len(train_set)} crops ({len(x)} with augmentation), "
          f"{len(test)} held out")
    weights, biases, classes = fit_svm(x, y, cfg)
    save_model(out_path, weights, biases, classes, cfg)
    report(SignModel(hog, weights, biases, classes), test, hog, cfg)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Train the sign classifier (laptop, scikit-learn).")
    p.add_argument("data", help="folder with pause/, left/, right/, straight/, none/")
    p.add_argument("--holdout", default="", help="comma-separated recording names to test on")
    p.add_argument("--out", default=os.path.join(_HERE, config.SIGN_MODEL_PATH))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    holdout = [h for h in args.holdout.split(",") if h]
    train(args.data, holdout, args.out, config, args.seed)


if __name__ == "__main__":
    main()
