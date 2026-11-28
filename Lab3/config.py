from pathlib import Path
from typing import Any

# Фіксований порядок класів (індекси 0-5)
CLASSES: list[str] = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]
CLASS_TO_IDX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

# Параметри зображень та нормалізації ImageNet
IMG_SIZE: int = 160
CROP_SCALE: tuple[float, float] = (0.7, 1.0)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Параметри поділу вибірки
SPLIT_SEED: int = 0
TRAIN_SAMPLES_PER_CLASS: int = 200

# Гіперпараметри навчання та завантажувачів
BATCH_SIZE_TRAIN: int = 32
BATCH_SIZE_EVAL: int = 64
NUM_WORKERS: int = 0
EPOCHS: int = 8
WEIGHT_DECAY: float = 1e-4

# Конфігурація трьох режимів навчання
REGIMES: dict[str, dict[str, Any]] = {
    "scratch": {
        "pretrained": False,
        "trainable_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
        "eval_modules": [],
        "lr_groups": [
            {"modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"], "lr": 3e-3}
        ],
        "expected_trainable_params": 11_179_590,
    },
    "frozen": {
        "pretrained": True,
        "trainable_modules": ["fc"],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        "lr_groups": [
            {"modules": ["fc"], "lr": 3e-3}
        ],
        "expected_trainable_params": 3_078,
    },
    "unfreeze": {
        "pretrained": True,
        "trainable_modules": ["layer4", "fc"],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "lr_groups": [
            {"modules": ["fc"], "lr": 3e-3},
            {"modules": ["layer4"], "lr": 3e-4},
        ],
        "expected_trainable_params": 8_396_806,
    },
}

# Контрольна таблиця поділу: (кількість_файлів, сума_числових_імен)
EXPECTED_SPLIT_STATS: dict[str, dict[str, tuple[int, int]]] = {
    "buildings": {
        "train": (200, 1_996_761),
        "validation": (218, 4_837_323),
        "test": (219, 4_837_559),
    },
    "forest": {
        "train": (200, 2_032_625),
        "validation": (237, 5_276_962),
        "test": (237, 5_291_326),
    },
    "glacier": {
        "train": (200, 2_183_923),
        "validation": (276, 6_135_652),
        "test": (277, 6_157_250),
    },
    "mountain": {
        "train": (200, 1_950_345),
        "validation": (262, 5_789_535),
        "test": (263, 5_849_765),
    },
    "sea": {
        "train": (200, 1_987_613),
        "validation": (255, 5_655_068),
        "test": (255, 5_668_011),
    },
    "street": {
        "train": (200, 1_841_817),
        "validation": (250, 5_541_845),
        "test": (251, 5_547_533),
    },
}

LAB3_DIR: Path = Path(__file__).resolve().parent

DEFAULT_SPLIT_PATH: Path = LAB3_DIR / "split.csv"
DEFAULT_RUNS_DIR: Path = LAB3_DIR / "runs"
DEFAULT_FIGURES_DIR: Path = LAB3_DIR / "figures"
DEFAULT_MODEL_PATH: Path = LAB3_DIR / "lab3_model.pt"