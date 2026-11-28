import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Lab3.config import (
    CLASSES,
    CLASS_TO_IDX,
    IMG_SIZE,
    CROP_SCALE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLIT_SEED,
    TRAIN_SAMPLES_PER_CLASS,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_EVAL,
    NUM_WORKERS,
)


def _resolve_split_dirs(data_dir: Path) -> tuple[Path, Path]:
    """Знаходить коректні шляхи до seg_train та seg_test."""
    train_dir = data_dir / "seg_train" / "seg_train"
    if not train_dir.exists():
        train_dir = data_dir / "seg_train"

    test_dir = data_dir / "seg_test" / "seg_test"
    if not test_dir.exists():
        test_dir = data_dir / "seg_test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Не знайдено теки seg_train або seg_test у директорії {data_dir}"
        )
    return train_dir, test_dir


def build_split(data_dir: str | Path, output_csv: str | Path = "split.csv") -> Path:
    """
    Формує детермінований поділ на train, validation і test та зберігає у split.csv.
    """
    data_path = Path(data_dir)
    out_path = Path(output_csv)
    train_dir, test_dir = _resolve_split_dirs(data_path)

    rows: list[tuple[str, str, str]] = []

    # 1. Формування train: один генератор для всіх класів по черзі
    rng_train = np.random.default_rng(SPLIT_SEED)
    for cls_name in CLASSES:
        cls_dir = train_dir / cls_name
        files = sorted(f.name for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg")
        idx = np.arange(len(files))
        rng_train.shuffle(idx)
        for i in idx[:TRAIN_SAMPLES_PER_CLASS]:
            rows.append((files[i], cls_name, "train"))

    # 2. Формування validation і test: новий генератор для всіх класів по черзі
    rng_eval = np.random.default_rng(SPLIT_SEED)
    for cls_name in CLASSES:
        cls_dir = test_dir / cls_name
        files = sorted(f.name for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg")
        idx = np.arange(len(files))
        rng_eval.shuffle(idx)
        cut = len(files) // 2
        for i in idx[:cut]:
            rows.append((files[i], cls_name, "validation"))
        for i in idx[cut:]:
            rows.append((files[i], cls_name, "test"))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "part"])
        writer.writerows(rows)

    return out_path


def get_transforms(is_train: bool) -> transforms.Compose:
    """Повертає конвеєр перетворень для train або validation/test."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class IntelImageDataset(Dataset):
    """Dataset для завантаження зображень суворо за файлом split.csv."""

    def __init__(
        self,
        data_dir: str | Path,
        split_csv: str | Path,
        part: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        if part not in {"train", "validation", "test"}:
            raise ValueError(f"Невідома частина вибірки: {part}")

        self.part = part
        self.transform = transform
        train_dir, test_dir = _resolve_split_dirs(Path(data_dir))
        self.base_dir = train_dir if part == "train" else test_dir

        self.samples: list[tuple[str, int]] = []
        with Path(split_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["part"] == part:
                    filename = row["file"]
                    label = row["label"]
                    self.samples.append((filename, CLASS_TO_IDX[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        filename, class_idx = self.samples[idx]
        cls_name = CLASSES[class_idx]
        img_path = self.base_dir / cls_name / filename

        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")

        tensor = self.transform(img_rgb) if self.transform else transforms.ToTensor()(img_rgb)
        return tensor, class_idx, filename


def get_dataloaders(
    data_dir: str | Path,
    split_csv: str | Path = "split.csv",
    include_test: bool = False,
) -> dict[str, DataLoader]:
    """
    Створює DataLoader-и. За замовчуванням test не завантажується.
    """
    loaders: dict[str, DataLoader] = {}

    train_ds = IntelImageDataset(
        data_dir=data_dir,
        split_csv=split_csv,
        part="train",
        transform=get_transforms(is_train=True),
    )
    loaders["train"] = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    val_ds = IntelImageDataset(
        data_dir=data_dir,
        split_csv=split_csv,
        part="validation",
        transform=get_transforms(is_train=False),
    )
    loaders["validation"] = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if include_test:
        test_ds = IntelImageDataset(
            data_dir=data_dir,
            split_csv=split_csv,
            part="test",
            transform=get_transforms(is_train=False),
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE_EVAL,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    return loaders