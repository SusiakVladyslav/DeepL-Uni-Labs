import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
from torchvision.models import resnet18

from Lab3.config import (
    CLASSES,
    DEFAULT_FIGURES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_SPLIT_PATH,
    IMG_SIZE,
)
from Lab3.data import _resolve_split_dirs, get_dataloaders
from Lab3.train_cli import EXPERIMENT_PLAN


def _compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_runs(runs_dir: Path, split_csv: Path) -> list[dict]:
    """Завантажує всі 7 журналів run.json та перевіряє збіг SHA-256 split.csv."""
    if not split_csv.exists():
        raise FileNotFoundError(f"Файл поділу {split_csv} не знайдено!")

    current_sha256 = _compute_file_sha256(split_csv)
    runs: list[dict] = []

    for regime, seed in EXPERIMENT_PLAN:
        run_path = runs_dir / f"{regime}_seed{seed}" / "run.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Відсутній журнал запуску: {run_path}")

        with run_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("split_sha256") != current_sha256:
            raise ValueError(
                f"Невідповідність SHA-256 split.csv у {run_path}! "
                f"Очікувалось {current_sha256}, у журналі {data.get('split_sha256')}."
            )
        runs.append(data)

    print("Перевірка SHA-256 split.csv у всіх 7 журналах: УСПІШНО")
    return runs


def plot_training_curves(runs: list[dict], out_path: Path) -> None:
    """Будує рисунок із двох панелей (train loss та val macro-F1) для всіх 7 запусків."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"scratch": "#d62728", "frozen": "#1f77b4", "unfreeze": "#2ca02c"}
    linestyles = {1: "-", 2: "--", 3: ":"}

    for run in runs:
        regime = run["regime"]
        seed = run["seed"]
        epochs = list(range(1, len(run["train_loss_per_epoch"]) + 1))
        label = f"{regime} (seed={seed})"

        ax_loss.plot(
            epochs,
            run["train_loss_per_epoch"],
            label=label,
            color=colors[regime],
            linestyle=linestyles[seed],
            marker="o",
            markersize=4,
        )
        ax_f1.plot(
            epochs,
            run["val_macro_f1_per_epoch"],
            label=label,
            color=colors[regime],
            linestyle=linestyles[seed],
            marker="o",
            markersize=4,
        )

    ax_loss.set_title("Втрата на train за епохами")
    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_xticks(range(1, 9))
    ax_loss.grid(True, alpha=0.3)

    ax_f1.set_title("macro-F1 на validation за епохами")
    ax_f1.set_xlabel("Епоха")
    ax_f1.set_ylabel("macro-F1")
    ax_f1.set_xticks(range(1, 9))
    ax_f1.grid(True, alpha=0.3)
    ax_f1.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Криві навчання збережено у {out_path}")


def decompose_effects(runs: list[dict]) -> tuple[str, int, float, dict[str, float]]:
    """
    Виводить таблицю 7 запусків, обчислює середні, розмахи R, ефекти Delta_pre і Delta_unf,
    та визначає очікуваний переможець за правилом вибору.
    """
    print("\n=== ТАБЛИЦЯ 7 ЗАПУСКІВ (Пункт 3) ===")
    header = f"{'#':<3} | {'Режим':<10} | {'Зерно':<6} | {'Втрата train (еп. 8)':<22} | {'macro-F1 val (еп. 8)':<22} | {'Час, с':<10}"
    print(header)
    print("-" * len(header))

    by_regime: dict[str, list[dict]] = {"scratch": [], "frozen": [], "unfreeze": []}
    for idx, run in enumerate(runs, start=1):
        by_regime[run["regime"]].append(run)
        print(
            f"{idx:<3} | {run['regime']:<10} | {run['seed']:<6} | "
            f"{run['final_train_loss']:<22.4f} | {run['final_val_macro_f1']:<22.4f} | "
            f"{run['training_time_sec']:<10.2f}"
        )

    f_scratch_1 = by_regime["scratch"][0]["final_val_macro_f1"]
    frozen_scores = [r["final_val_macro_f1"] for r in by_regime["frozen"]]
    unfreeze_scores = [r["final_val_macro_f1"] for r in by_regime["unfreeze"]]

    mean_frozen = float(np.mean(frozen_scores))
    range_frozen = float(np.max(frozen_scores) - np.min(frozen_scores))

    mean_unfreeze = float(np.mean(unfreeze_scores))
    range_unfreeze = float(np.max(unfreeze_scores) - np.min(unfreeze_scores))

    noise_r = max(range_frozen, range_unfreeze)
    delta_pre = mean_frozen - f_scratch_1
    delta_unf = mean_unfreeze - mean_frozen

    print("\n=== РОЗКЛАД ЕФЕКТІВ (Пункт 4) ===")
    print(f"F_scratch,1 : {f_scratch_1:.4f}")
    print(f"F_frozen    : {mean_frozen:.4f} (розмах R_frozen = {range_frozen:.4f})")
    print(f"F_unfreeze  : {mean_unfreeze:.4f} (розмах R_unfreeze = {range_unfreeze:.4f})")
    print(f"Поріг шуму R: {noise_r:.4f}")
    print(
        f"Delta_pre   : {delta_pre:+.4f} "
        f"({'|Delta| > R (варте перевірки)' if abs(delta_pre) > noise_r else '|Delta| <= R (не відрізняється від розкиду)'})"
    )
    print(
        f"Delta_unf   : {delta_unf:+.4f} "
        f"({'|Delta| > R (варте перевірки)' if abs(delta_unf) > noise_r else '|Delta| <= R (не відрізняється від розкиду)'})"
    )

    regime_means = {
        "scratch": f_scratch_1,
        "frozen": mean_frozen,
        "unfreeze": mean_unfreeze,
    }
    best_regime = max(regime_means.keys(), key=lambda r: regime_means[r])
    best_run = max(by_regime[best_regime], key=lambda r: r["final_val_macro_f1"])

    stats = {
        "R": noise_r,
        "delta_pre": delta_pre,
        "delta_unf": delta_unf,
        "best_val_f1": best_run["final_val_macro_f1"],
    }
    return best_regime, best_run["seed"], best_run["final_val_macro_f1"], stats


def evaluate_on_test_cpu(
    model_path: Path,
    expected_regime: str,
    expected_seed: int,
    data_dir: Path,
    split_csv: Path,
    figures_dir: Path,
) -> None:
    """
    Перевіряє відповідність lab3_model.pt правилу вибору, один раз обчислює логіти на test (CPU),
    будує матрицю плутанини та виводить 5 характерних помилок.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("regime") != expected_regime or checkpoint.get("seed") != expected_seed:
        raise ValueError(
            f"Файл {model_path} містить ({checkpoint.get('regime')}, seed={checkpoint.get('seed')}), "
            f"що не відповідає правилу вибору ({expected_regime}, seed={expected_seed})!"
        )
    if checkpoint.get("classes") != CLASSES or checkpoint.get("img_size") != IMG_SIZE:
        raise ValueError("Невідповідність метаданих classes або img_size у lab3_model.pt!")

    print(
        f"\n=== ТЕСТОВА ОЦІНКА НА CPU (Пункт 5): модель {expected_regime} (seed={expected_seed}) ==="
    )

    cpu_device = torch.device("cpu")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(cpu_device)
    model.eval()

    loaders = get_dataloaders(data_dir=data_dir, split_csv=split_csv, include_test=True)
    test_loader = loaders["test"]

    all_logits: list[torch.Tensor] = []
    all_targets: list[int] = []
    all_filenames: list[str] = []

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            logits = model(images.to(cpu_device))
            all_logits.append(logits.cpu())
            all_targets.extend(int(y) for y in labels)
            all_filenames.extend(filenames)

    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = torch.softmax(logits_tensor, dim=-1)
    preds_arr = torch.argmax(logits_tensor, dim=-1).numpy()
    targets_arr = np.array(all_targets)

    test_macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro"))
    accuracy = float(np.mean(preds_arr == targets_arr))
    n_test = len(targets_arr)
    se_acc = math.sqrt(accuracy * (1.0 - accuracy) / n_test)

    print(f"test macro-F1 : {test_macro_f1:.4f}")
    print(f"test accuracy : {accuracy:.4f} (SE = {se_acc:.4f} для n={n_test})")

    # Матриця плутанини 6x6
    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(len(CLASSES))))
    print("\n--- Матриця плутанини (рядки — справжній клас, стовпці — передбачений) ---")
    cm_header = f"{'':<12} | " + " | ".join(f"{c:<9}" for c in CLASSES)
    print(cm_header)
    print("-" * len(cm_header))
    for i, cls_name in enumerate(CLASSES):
        row_str = " | ".join(f"{cm[i, j]:<9d}" for j in range(len(CLASSES)))
        print(f"{cls_name:<12} | {row_str}")

    # Збереження теплової карти матриці плутанини
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(CLASSES)))
    ax_cm.set_yticks(range(len(CLASSES)))
    ax_cm.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax_cm.set_yticklabels(CLASSES)
    ax_cm.set_xlabel("Передбачений клас")
    ax_cm.set_ylabel("Справжній клас")
    ax_cm.set_title(f"Матриця плутанини на test (macro-F1 = {test_macro_f1:.4f})")

    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax_cm.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig_cm.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_fig_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=150)
    plt.close(fig_cm)
    print(f"Рисунок матриці плутанини збережено у {cm_fig_path}")

    # П'ять характерних помилок
    off_diag_pairs: list[tuple[int, int, int]] = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                off_diag_pairs.append((int(cm[i, j]), i, j))

    # Сортування за спаданням частоти, далі за зростанням i, потім j
    off_diag_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    # Групування помилкових прикладів за парою (true_idx, pred_idx), відсортованих за іменем файлу
    errors_by_pair: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for idx, (t_idx, p_idx) in enumerate(zip(targets_arr, preds_arr)):
        if t_idx != p_idx:
            pair = (int(t_idx), int(p_idx))
            pred_prob = float(probs_tensor[idx, p_idx].item())
            errors_by_pair.setdefault(pair, []).append((all_filenames[idx], pred_prob))

    for pair in errors_by_pair:
        errors_by_pair[pair].sort(key=lambda item: item[0])

    selected_errors: list[tuple[str, str, str, float, int]] = []
    pair_cursors: dict[tuple[int, int], int] = { (i, j): 0 for _, i, j in off_diag_pairs }

    while len(selected_errors) < 5 and off_diag_pairs:
        added_in_round = False
        for count, true_idx, pred_idx in off_diag_pairs:
            if len(selected_errors) >= 5:
                break
            pair = (true_idx, pred_idx)
            cursor = pair_cursors[pair]
            candidates = errors_by_pair.get(pair, [])
            if cursor < len(candidates):
                fname, prob = candidates[cursor]
                pair_cursors[pair] += 1
                selected_errors.append(
                    (fname, CLASSES[true_idx], CLASSES[pred_idx], prob, count)
                )
                added_in_round = True
        if not added_in_round:
            break

    print("\n--- П'ять характерних помилок на test ---")
    err_header = (
        f"{'Файл':<14} | {'Справжній клас':<15} | {'Передбачений':<15} | "
        f"{'Імовірність':<13} | {'Кількість у test':<16}"
    )
    print(err_header)
    print("-" * len(err_header))
    for fname, true_cls, pred_cls, prob, count in selected_errors:
        print(
            f"{fname:<14} | {true_cls:<15} | {pred_cls:<15} | "
            f"{prob:<13.4f} | {count:<16d}"
        )

    # Візуалізація 5 помилкових зображень
    _, test_dir = _resolve_split_dirs(data_dir)
    fig_err, axes = plt.subplots(1, len(selected_errors), figsize=(3.5 * len(selected_errors), 4))
    if len(selected_errors) == 1:
        axes = [axes]

    for ax, (fname, true_cls, pred_cls, prob, _) in zip(axes, selected_errors):
        img_path = test_dir / true_cls / fname
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"{fname}\nTrue: {true_cls}\nPred: {pred_cls} ({prob:.2f})",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    err_fig_path = figures_dir / "top5_errors.png"
    plt.savefig(err_fig_path, dpi=150)
    plt.close(fig_err)
    print(f"Рисунок п'яти помилок збережено у {err_fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР3: Аналіз результатів і одноразова оцінка на test (CPU)")
    parser.add_argument("--data-dir", type=Path, required=True, help="Шлях до розпакованого набору даних")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_PATH, help="Шлях до split.csv")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Директорія журналів запусків")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Директорія для рисунків")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Шлях до вибраної моделі")
    args = parser.parse_args()

    runs = load_and_validate_runs(runs_dir=args.runs_dir, split_csv=args.split_csv)
    plot_training_curves(runs=runs, out_path=args.figures_dir / "training_curves.png")
    best_regime, best_seed, _, _ = decompose_effects(runs=runs)

    evaluate_on_test_cpu(
        model_path=args.model_path,
        expected_regime=best_regime,
        expected_seed=best_seed,
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()