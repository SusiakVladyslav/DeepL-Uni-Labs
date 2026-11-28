import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torchvision

from Lab3.checks import (
    check_initial_loss,
    save_batch_preview,
    verify_regimes_and_bn_bug,
    verify_split,
)
from Lab3.config import (
    BATCH_SIZE_EVAL,
    BATCH_SIZE_TRAIN,
    CLASSES,
    CROP_SCALE,
    DEFAULT_FIGURES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_SPLIT_PATH,
    EPOCHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_WORKERS,
    REGIMES,
    SPLIT_SEED,
    TRAIN_SAMPLES_PER_CLASS,
    WEIGHT_DECAY,
)
from Lab3.data import build_split, get_dataloaders
from Lab3.modeling import build_model_and_optimizer, get_device, set_train_mode

EXPERIMENT_PLAN: list[tuple[str, int]] = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def _get_git_metadata() -> dict[str, str | bool]:
    """Збирає хеш поточного коміту та прапорець незакомічених змін."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"git_commit": commit_hash, "git_dirty": bool(status_out)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": True}


def _compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Обчислює macro-F1 на validation та повертає матрицю логітів (1498 x 6)."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    preds_arr = np.argmax(logits_arr, axis=1)

    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro"))
    return macro_f1, logits_arr


def run_single_experiment(
    regime: str,
    seed: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_sha256: str,
    git_meta: dict[str, str | bool],
    runs_dir: Path,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Виконує один повний цикл навчання (8 епох) та зберігає run.json і val_logits.npy."""
    model, optimizer = build_model_and_optimizer(regime=regime, seed=seed, device=device)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_macro_f1s: list[float] = []
    total_train_samples = len(train_loader.dataset)

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        running_loss_sum = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss_sum += float(loss.item()) * len(labels)
            loss.backward()
            optimizer.step()

        epoch_train_loss = running_loss_sum / total_train_samples
        epoch_val_f1, val_logits = _evaluate_validation(model, val_loader, device)

        train_losses.append(epoch_train_loss)
        val_macro_f1s.append(epoch_val_f1)

        print(
            f"[{regime:<8} | seed={seed}] Епоха {epoch}/{EPOCHS} — "
            f"train loss: {epoch_train_loss:.4f}, val macro-F1: {epoch_val_f1:.4f}"
        )

    elapsed_sec = time.perf_counter() - start_time

    run_folder = runs_dir / f"{regime}_seed{seed}"
    run_folder.mkdir(parents=True, exist_ok=True)

    np.save(run_folder / "val_logits.npy", val_logits.astype(np.float32))

    run_record = {
        "regime": regime,
        "seed": seed,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "crop_scale": list(CROP_SCALE),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "split_seed": SPLIT_SEED,
        "train_samples_per_class": TRAIN_SAMPLES_PER_CLASS,
        "batch_size_train": BATCH_SIZE_TRAIN,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "regime_config": REGIMES[regime],
        "split_sha256": split_sha256,
        "git_commit": git_meta["git_commit"],
        "git_dirty": git_meta["git_dirty"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "train_loss_per_epoch": train_losses,
        "val_macro_f1_per_epoch": val_macro_f1s,
        "final_train_loss": train_losses[-1],
        "final_val_macro_f1": val_macro_f1s[-1],
        "training_time_sec": round(elapsed_sec, 3),
    }

    with (run_folder / "run.json").open("w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    return val_macro_f1s[-1], cpu_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР3: Перевірки протоколу та 7 запусків навчання")
    parser.add_argument("--data-dir", type=Path, required=True, help="Шлях до розпакованого набору даних")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_PATH, help="Шлях до split.csv")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Директорія журналів запусків")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Директорія для рисунків")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Файл вибраної моделі")
    args = parser.parse_args()

    device = get_device()
    print(f"Використовується пристрій навчання: {device}")

    # 1. Формування та перевірка поділу
    build_split(data_dir=args.data_dir, output_csv=args.split_csv)
    verify_split(args.split_csv)
    split_sha256 = _compute_file_sha256(args.split_csv)

    # Завантаження лише train та validation (test строго заблоковано)
    loaders = get_dataloaders(data_dir=args.data_dir, split_csv=args.split_csv, include_test=False)
    train_loader = loaders["train"]
    val_loader = loaders["validation"]

    # 2. Перевірки до навчання та верифікація режимів (пункти 1 і 2)
    save_batch_preview(train_loader, args.figures_dir / "train_batch_preview.png")
    check_initial_loss(train_loader, device)
    verify_regimes_and_bn_bug(train_loader, device)

    # 3. Основний експеримент — 7 запусків (пункт 3)
    print("\n=== ЕТАП 3: Запуск 7 експериментів за протоколом ===")
    git_meta = _get_git_metadata()
    results_by_regime: dict[str, list[tuple[int, float, dict[str, torch.Tensor]]]] = {
        "scratch": [],
        "frozen": [],
        "unfreeze": [],
    }

    for idx, (regime, seed) in enumerate(EXPERIMENT_PLAN, start=1):
        print(f"\n--- Запуск {idx}/7: режим={regime}, seed={seed} ---")
        final_f1, state_dict = run_single_experiment(
            regime=regime,
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            split_sha256=split_sha256,
            git_meta=git_meta,
            runs_dir=args.runs_dir,
        )
        results_by_regime[regime].append((seed, final_f1, state_dict))

    # 4. Вибір конфігурації за validation (пункт 5, перша частина)
    regime_means: dict[str, float] = {
        r: float(np.mean([item[1] for item in items]))
        for r, items in results_by_regime.items()
    }
    best_regime = max(regime_means.keys(), key=lambda r: regime_means[r])
    best_seed, best_f1, best_state_dict = max(
        results_by_regime[best_regime], key=lambda item: item[1]
    )

    checkpoint = {
        "state_dict": best_state_dict,
        "regime": best_regime,
        "seed": best_seed,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
    }
    torch.save(checkpoint, args.model_out)

    print("\n=== ВИБІР КОНФІГУРАЦІЇ ЗА VALIDATION ЗАВЕРШЕНО ===")
    for r, mean_val in regime_means.items():
        print(f"Середній macro-F1 ({r}): {mean_val:.4f}")
    print(
        f"Обрано режим '{best_regime}' (seed={best_seed}, val macro-F1={best_f1:.4f}) -> "
        f"збережено у {args.model_out}"
    )


if __name__ == "__main__":
    main()