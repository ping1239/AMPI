from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "stft_image"
SAVE_DIR = PROJECT_DIR / "dataset"
WINDOW_SIZE = 128
STEP_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42


def create_windows(orig, diff, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Pair aligned orig/diff windows and concatenate them into 12 channels."""
    if orig.shape != diff.shape:
        raise ValueError(f"orig/diff shape mismatch: {orig.shape} != {diff.shape}")
    if orig.ndim != 3 or orig.shape[:2] != (6, 65):
        raise ValueError(f"Expected (6, 65, time), got {orig.shape}")

    windows = []
    for start in range(0, orig.shape[2] - window_size + 1, step_size):
        end = start + window_size
        windows.append(np.concatenate((orig[:, :, start:end], diff[:, :, start:end]), axis=0))
    return windows


def split_timeline(orig, diff):
    """Split a paired session before windowing to prevent overlapping-window leakage."""
    time_length = orig.shape[2]
    train_end = int(time_length * TRAIN_RATIO)
    val_end = int(time_length * (TRAIN_RATIO + VAL_RATIO))
    boundaries = ((0, train_end), (train_end, val_end), (val_end, time_length))
    return [(orig[:, :, start:end], diff[:, :, start:end]) for start, end in boundaries]


def find_session_pairs(class_dir):
    pairs = []
    for orig_path in sorted(class_dir.glob("*_orig.npy")):
        session_name = orig_path.name.removesuffix("_orig.npy")
        diff_path = class_dir / f"{session_name}_diff.npy"
        if not diff_path.is_file():
            raise FileNotFoundError(f"Missing diff file for {orig_path}: {diff_path}")
        pairs.append((session_name, orig_path, diff_path))
    if not pairs:
        raise FileNotFoundError(f"No *_orig.npy files found in {class_dir}")
    return pairs


def normalize_from_train(train, val, test):
    """Normalize every orig/diff channel using training-set statistics only."""
    mean = train.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    shape = (1, -1, 1, 1)
    return (
        ((train - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32),
        ((val - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32),
        ((test - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32),
        mean,
        std,
    )


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"STFT image directory not found: {SOURCE_DIR}")

    class_dirs = sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir())
    if len(class_dirs) != 5:
        raise ValueError(f"Expected exactly 5 class directories, found {len(class_dirs)}")

    class_names = np.array([path.name for path in class_dirs])
    split_features = [[], [], []]
    split_labels = [[], [], []]
    per_class_counts = np.zeros((3, len(class_dirs)), dtype=np.int64)

    print("Classes:", ", ".join(f"{index}={name}" for index, name in enumerate(class_names)))
    for label, class_dir in enumerate(class_dirs):
        for session_name, orig_path, diff_path in find_session_pairs(class_dir):
            orig = np.load(orig_path).astype(np.float32, copy=False)
            diff = np.load(diff_path).astype(np.float32, copy=False)
            if orig.shape != diff.shape:
                raise ValueError(f"{class_dir.name}/{session_name}: shape mismatch")
            if not np.isfinite(orig).all() or not np.isfinite(diff).all():
                raise ValueError(f"{class_dir.name}/{session_name}: NaN or Inf detected")

            for split_index, (orig_split, diff_split) in enumerate(split_timeline(orig, diff)):
                windows = create_windows(orig_split, diff_split)
                split_features[split_index].extend(windows)
                split_labels[split_index].extend([label] * len(windows))
                per_class_counts[split_index, label] += len(windows)

    rng = np.random.default_rng(RANDOM_SEED)
    arrays = []
    for features, labels in zip(split_features, split_labels):
        if not features:
            raise ValueError("One of train/validation/test splits has no windows")
        x = np.stack(features).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        order = rng.permutation(len(y))
        arrays.append((x[order], y[order]))

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = arrays
    x_train, x_val, x_test, mean, std = normalize_from_train(x_train, x_val, x_test)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, x, y in (
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ):
        np.save(SAVE_DIR / f"X_{split_name}.npy", x)
        np.save(SAVE_DIR / f"y_{split_name}.npy", y)
    np.save(SAVE_DIR / "classes.npy", class_names)
    np.savez(SAVE_DIR / "normalization.npz", mean=mean, std=std)

    for split_index, split_name in enumerate(("train", "val", "test")):
        details = ", ".join(
            f"{name}={per_class_counts[split_index, label]}"
            for label, name in enumerate(class_names)
        )
        print(f"{split_name}: total={per_class_counts[split_index].sum()} ({details})")
    print(f"Saved dataset to {SAVE_DIR}")


if __name__ == "__main__":
    main()
