from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "stft_image"
SAVE_DIR = PROJECT_DIR / "dataset"
DUAL_SAVE_DIR = SAVE_DIR / "dual"
COMBINED_SAVE_DIR = SAVE_DIR / "combined"
WINDOW_SIZE = 128
STEP_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42
CLASS_LABELS = (
    ("01_", "01_ Rapid Acceleration/Deceleration"),
    ("02_", "02_ Overload"),
    ("03_", "03_ Collision"),
    ("05_", "05_ Axis Load"),
    ("06_", "06_ Normal"),
)


def create_windows(orig, diff, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Create aligned 6-channel orig/diff window pairs."""
    if orig.shape != diff.shape:
        raise ValueError(f"orig/diff shape mismatch: {orig.shape} != {diff.shape}")
    if orig.ndim != 3 or orig.shape[:2] != (6, 65):
        raise ValueError(f"Expected (6, 65, time), got {orig.shape}")

    windows = []
    for start in range(0, orig.shape[2] - window_size + 1, step_size):
        end = start + window_size
        windows.append((orig[:, :, start:end], diff[:, :, start:end]))
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

    class_dirs = [SOURCE_DIR / directory_name for directory_name, _ in CLASS_LABELS]
    missing_dirs = [path.name for path in class_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing class directories: {missing_dirs}")

    class_names = np.array([class_name for _, class_name in CLASS_LABELS])
    split_orig_features = [[], [], []]
    split_diff_features = [[], [], []]
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
                split_orig_features[split_index].extend(window[0] for window in windows)
                split_diff_features[split_index].extend(window[1] for window in windows)
                split_labels[split_index].extend([label] * len(windows))
                per_class_counts[split_index, label] += len(windows)

    rng = np.random.default_rng(RANDOM_SEED)
    arrays = []
    for orig_features, diff_features, labels in zip(
        split_orig_features, split_diff_features, split_labels
    ):
        if not orig_features:
            raise ValueError("One of train/validation/test splits has no windows")
        x_orig = np.stack(orig_features).astype(np.float32)
        x_diff = np.stack(diff_features).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        order = rng.permutation(len(y))
        arrays.append((x_orig[order], x_diff[order], y[order]))

    (orig_train, diff_train, y_train), (orig_val, diff_val, y_val), (orig_test, diff_test, y_test) = arrays
    orig_train, orig_val, orig_test, orig_mean, orig_std = normalize_from_train(
        orig_train, orig_val, orig_test
    )
    diff_train, diff_val, diff_test, diff_mean, diff_std = normalize_from_train(
        diff_train, diff_val, diff_test
    )

    DUAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, x_orig, x_diff, y in (
        ("train", orig_train, diff_train, y_train),
        ("val", orig_val, diff_val, y_val),
        ("test", orig_test, diff_test, y_test),
    ):
        np.save(DUAL_SAVE_DIR / f"X_orig_{split_name}.npy", x_orig)
        np.save(DUAL_SAVE_DIR / f"X_diff_{split_name}.npy", x_diff)
        np.save(DUAL_SAVE_DIR / f"y_{split_name}.npy", y)
        np.save(COMBINED_SAVE_DIR / f"X_{split_name}.npy", np.concatenate((x_orig, x_diff), axis=1))
        np.save(COMBINED_SAVE_DIR / f"y_{split_name}.npy", y)
    np.save(DUAL_SAVE_DIR / "classes.npy", class_names)
    np.save(COMBINED_SAVE_DIR / "classes.npy", class_names)
    np.savez(
        DUAL_SAVE_DIR / "normalization.npz",
        orig_mean=orig_mean,
        orig_std=orig_std,
        diff_mean=diff_mean,
        diff_std=diff_std,
    )
    np.savez(
        COMBINED_SAVE_DIR / "normalization.npz",
        mean=np.concatenate((orig_mean, diff_mean)),
        std=np.concatenate((orig_std, diff_std)),
    )

    for split_index, split_name in enumerate(("train", "val", "test")):
        details = ", ".join(
            f"{name}={per_class_counts[split_index, label]}"
            for label, name in enumerate(class_names)
        )
        print(f"{split_name}: total={per_class_counts[split_index].sum()} ({details})")
    print(f"Saved dual dataset to {DUAL_SAVE_DIR}")
    print(f"Saved combined dataset to {COMBINED_SAVE_DIR}")


if __name__ == "__main__":
    main()
