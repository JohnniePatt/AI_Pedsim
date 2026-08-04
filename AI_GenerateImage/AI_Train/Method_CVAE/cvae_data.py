import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


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


class CVAEDensityMapDataset(Dataset):
    def __init__(self, dataset_root, subset: str, image_size: int, target_representation: str = "bw"):
        self.dataset_root = pathlib.Path(dataset_root)
        self.subset = subset
        self.image_size = int(image_size)
        self.target_representation = str(target_representation).lower()
        self.dir_a, self.dir_b, self.pairs = list_pair_files(self.dataset_root, subset, self.image_size)
        if not self.pairs:
            raise RuntimeError(f"[DATASET] {subset} split is empty at {self.dir_a}")

    @property
    def target_channels(self):
        return 3 if self.target_representation in {"color", "colorjet", "rgb", "jet"} else 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_path = self.pairs[index]
        b_path = self.dir_b / a_path.name
        a = Image.open(a_path).convert("RGB").resize((self.image_size, self.image_size), BILINEAR)
        a_arr = np.asarray(a, dtype=np.float32) / 255.0
        a_tensor = torch.from_numpy(a_arr).permute(2, 0, 1).contiguous()
        b_tensor = load_density_target(b_path, self.image_size, self.target_representation)
        return a_tensor, b_tensor, a_path.name


def make_density_dataset(
    dataset_root,
    subset,
    batch_size,
    image_size,
    shuffle,
    target_representation="bw",
    seed=42,
    num_workers=0,
):
    dataset = CVAEDensityMapDataset(dataset_root, subset, image_size, target_representation=target_representation)
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


def load_image(path, image_size, method="bicubic"):
    img = Image.open(path)
    orig_w, orig_h = img.size
    img = img.convert("RGB").resize((int(image_size), int(image_size)), BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor, orig_w, orig_h


def load_density_target(path, image_size, target_representation="bw"):
    target_representation = str(target_representation).lower()
    if target_representation in {"color", "colorjet", "rgb", "jet"}:
        img = Image.open(path).convert("RGB").resize((int(image_size), int(image_size)), BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    img = Image.open(path).convert("L").resize((int(image_size), int(image_size)), BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None, ...]).contiguous()
