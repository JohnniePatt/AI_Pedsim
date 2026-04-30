import pathlib

import tensorflow as tf


def list_pair_files(dataset_root, subset, image_size):
    base_root = pathlib.Path(dataset_root)
    subset_aliases = {
        "train": ["train", "training"],
        "validation": ["validation", "val", "valid"],
        "test": ["test", "testing"],
    }
    candidate_splits = subset_aliases.get(subset, [subset])

    dir_a = None
    dir_b = None
    for split_name in candidate_splits:
        cand_a = base_root / "A" / split_name
        cand_b = base_root / "B" / split_name
        if cand_a.exists() and cand_b.exists():
            dir_a = cand_a
            dir_b = cand_b
            break

    if dir_a is None or dir_b is None:
        a_root = base_root / "A"
        b_root = base_root / "B"
        a_splits = sorted([p.name for p in a_root.iterdir()]) if a_root.exists() else []
        b_splits = sorted([p.name for p in b_root.iterdir()]) if b_root.exists() else []
        raise FileNotFoundError(
            f"[DATASET-{subset}] split not found in {base_root}. "
            f"Tried splits: {candidate_splits}. "
            f"Available A splits: {a_splits if a_splits else 'A missing/empty'}, "
            f"B splits: {b_splits if b_splits else 'B missing/empty'}"
        )

    files_a = sorted([p for p in dir_a.glob("*.png")])
    files_b = sorted([p for p in dir_b.glob("*.png")])
    names_b = {p.name for p in files_b}
    pair_files = [p for p in files_a if p.name in names_b]

    print(f"[DATASET-{subset}] {len(pair_files)} images | resize -> {image_size}x{image_size}")
    return dir_a, dir_b, pair_files


def make_dataset(dataset_root, subset, batch_size, image_size, shuffle, seed=42):
    dir_a, dir_b, pair_files = list_pair_files(dataset_root, subset, image_size)
    if len(pair_files) == 0:
        raise RuntimeError(f"[DATASET] {subset} split is empty at {dir_a}")

    path_pairs = [(str(p), str(dir_b / p.name)) for p in pair_files]
    paths_a = [a for a, _ in path_pairs]
    paths_b = [b for _, b in path_pairs]

    def _load_pair(path_a, path_b):
        img_a = tf.io.read_file(path_a)
        img_a = tf.image.decode_png(img_a, channels=3)
        img_a = tf.image.resize(img_a, [image_size, image_size], method="bicubic")
        img_a = tf.cast(img_a, tf.float32) / 127.5 - 1.0

        img_b = tf.io.read_file(path_b)
        img_b = tf.image.decode_png(img_b, channels=3)
        # Keep thin trajectory structure sharp (avoid blur from bicubic on labels).
        img_b = tf.image.resize(img_b, [image_size, image_size], method="nearest")
        img_b = tf.cast(img_b, tf.float32) / 127.5 - 1.0

        return img_a, img_b

    ds = tf.data.Dataset.from_tensor_slices((paths_a, paths_b))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(path_pairs), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, path_pairs


def list_test_pairs(dataset_root):
    base_root = pathlib.Path(dataset_root)
    candidates = ["test", "testing"]

    dir_a = None
    dir_b = None
    for split in candidates:
        cand_a = base_root / "A" / split
        cand_b = base_root / "B" / split
        if cand_a.exists() and cand_b.exists():
            dir_a = cand_a
            dir_b = cand_b
            break

    if dir_a is None or dir_b is None:
        raise FileNotFoundError(f"No test split found under {base_root}")

    files_a = sorted([p for p in dir_a.glob("*.png")])
    names_b = {p.name for p in dir_b.glob("*.png")}
    pair_files = [p for p in files_a if p.name in names_b]
    return dir_a, dir_b, pair_files


def load_image(path, image_size, method="bicubic"):
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_png(raw, channels=3)
    orig_h = int(tf.shape(img)[0])
    orig_w = int(tf.shape(img)[1])
    img = tf.image.resize(img, [image_size, image_size], method=method)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img, orig_w, orig_h
