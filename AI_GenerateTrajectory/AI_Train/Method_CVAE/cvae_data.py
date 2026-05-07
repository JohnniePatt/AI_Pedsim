import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def list_pair_files(dataset_root, subset, image_size=None):
    base_root = pathlib.Path(dataset_root)
    aliases = {
        "train": ["train", "training"],
        "validation": ["validation", "val", "valid"],
        "test": ["test", "testing"],
    }
    candidate_splits = aliases.get(subset, [subset])
    dir_a = None
    dir_b = None
    for split in candidate_splits:
        cand_a = base_root / "A" / split
        cand_b = base_root / "B" / split
        if cand_a.exists() and cand_b.exists():
            dir_a = cand_a
            dir_b = cand_b
            break
    if dir_a is None or dir_b is None:
        raise FileNotFoundError(f"[DATASET-{subset}] split not found under {base_root}")

    a_files = sorted(dir_a.glob("*.png"), key=lambda p: p.name)
    b_names = {p.name for p in dir_b.glob("*.png")}
    pairs = [p for p in a_files if p.name in b_names]
    print(f"[DATASET-{subset}] {len(pairs)} images | resize -> {image_size}x{image_size}")
    return dir_a, dir_b, pairs


class CVAETrajectoryDataset(Dataset):
    def __init__(self, dataset_root, subset: str, image_size: int):
        self.dataset_root = pathlib.Path(dataset_root)
        self.subset = subset
        self.image_size = int(image_size)
        self.dir_a, self.dir_b, self.pairs = list_pair_files(self.dataset_root, subset, self.image_size)
        if not self.pairs:
            raise RuntimeError(f"[DATASET] {subset} split is empty at {self.dir_a}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_path = self.pairs[index]
        b_path = self.dir_b / a_path.name
        a = Image.open(a_path).convert("RGB").resize((self.image_size, self.image_size), BILINEAR)
        b = Image.open(b_path).convert("L").resize((self.image_size, self.image_size), NEAREST)
        a_arr = np.asarray(a, dtype=np.float32) / 255.0
        b_arr = (np.asarray(b, dtype=np.float32) >= 128).astype(np.float32)
        a_tensor = torch.from_numpy(a_arr).permute(2, 0, 1).contiguous()
        b_tensor = torch.from_numpy(b_arr[None, ...]).contiguous()
        return a_tensor, b_tensor, a_path.name


def make_dataset(dataset_root, subset, batch_size, image_size, shuffle, seed=42, num_workers=0):
    dataset = CVAETrajectoryDataset(dataset_root, subset, image_size)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=max(int(num_workers), 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )
    return loader, [(str(p), str(dataset.dir_b / p.name)) for p in dataset.pairs]


def list_test_pairs(dataset_root):
    return list_pair_files(dataset_root, "test", image_size="original")


def load_image(path, image_size, method="bicubic"):
    resample = NEAREST if method == "nearest" else BILINEAR
    img = Image.open(path)
    orig_w, orig_h = img.size
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((int(image_size), int(image_size)), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, orig_w, orig_h


def load_mask(path, image_size):
    img = Image.open(path).convert("L")
    orig_w, orig_h = img.size
    img = img.resize((int(image_size), int(image_size)), NEAREST)
    arr = (np.asarray(img, dtype=np.float32) >= 128).astype(np.float32)
    return torch.from_numpy(arr[None, ...]).contiguous(), orig_w, orig_h
