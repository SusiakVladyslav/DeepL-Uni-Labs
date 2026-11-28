import copy
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Lab3.config import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EXPECTED_SPLIT_STATS,
    REGIMES,
)
from Lab3.modeling import build_model_and_optimizer, set_train_mode

TOP_LEVEL_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def verify_split(split_csv: str | Path) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Перевіряє цілісність split.csv (відсутність перетинів між частинами)
    та звіряє кількість і суму числових імен файлів із контрольною таблицею.
    """
    seen_files: set[str] = set()
    stats: dict[str, dict[str, list[int]]] = {
        cls_name: {
            "train": [0, 0],
            "validation": [0, 0],
            "test": [0, 0],
        }
        for cls_name in CLASSES
    }

    with Path(split_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file"]
            label = row["label"]
            part = row["part"]

            if fname in seen_files:
                raise AssertionError(f"Файл {fname} зустрічається у split.csv більше одного разу!")
            seen_files.add(fname)

            file_num = int(Path(fname).stem)
            stats[label][part][0] += 1
            stats[label][part][1] += file_num

    print("\n=== ПЕРЕВІРКА 1.1: Контрольна таблиця поділу ===")
    header = f"{'Клас':<12} | {'train':<18} | {'validation':<18} | {'test':<18}"
    print(header)
    print("-" * len(header))

    all_match = True
    result_stats: dict[str, dict[str, tuple[int, int]]] = {}

    for cls_name in CLASSES:
        result_stats[cls_name] = {}
        row_cells = []
        for part in ("train", "validation", "test"):
            actual = (stats[cls_name][part][0], stats[cls_name][part][1])
            expected = EXPECTED_SPLIT_STATS[cls_name][part]
            result_stats[cls_name][part] = actual
            if actual != expected:
                all_match = False
            row_cells.append(f"{actual[0]}/{actual[1]:,}".replace(",", " "))
        print(f"{cls_name:<12} | {row_cells[0]:<18} | {row_cells[1]:<18} | {row_cells[2]:<18}")

    tot_train = (sum(stats[c]["train"][0] for c in CLASSES), sum(stats[c]["train"][1] for c in CLASSES))
    tot_val = (sum(stats[c]["validation"][0] for c in CLASSES), sum(stats[c]["validation"][1] for c in CLASSES))
    tot_test = (sum(stats[c]["test"][0] for c in CLASSES), sum(stats[c]["test"][1] for c in CLASSES))
    print("-" * len(header))
    print(
        f"{'Разом':<12} | "
        f"{tot_train[0]}/{tot_train[1]:,}".replace(",", " ").ljust(18) + " | "
        f"{tot_val[0]}/{tot_val[1]:,}".replace(",", " ").ljust(18) + " | "
        f"{tot_test[0]}/{tot_test[1]:,}".replace(",", " ").ljust(18)
    )
    print(f"Усі 18 клітинок збігаються з контрольною таблицею: {'ТАК' if all_match else 'НІ'}")

    if not all_match:
        raise AssertionError("Поділ split.csv не збігається з контрольною таблицею!")

    return result_stats


def save_batch_preview(train_loader: DataLoader, out_path: str | Path) -> None:
    """
    Зберігає рисунок із перших 16 зображень навчального пакета після аугментацій
    з оберненим нормуванням та підписаною міткою класу.
    """
    images, labels, _ = next(iter(train_loader))
    images = images[:16]
    labels = labels[:16]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        img_tensor = (images[idx].cpu() * std + mean).clamp(0.0, 1.0)
        img_np = np.transpose(img_tensor.numpy(), (1, 2, 0))
        ax.imshow(img_np)
        ax.set_title(CLASSES[int(labels[idx])], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n=== ПЕРЕВІРКА 1.2: Прев'ю пакета збережено у {out_path} ===")


def check_initial_loss(train_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """
    Обчислює початкову втрату L_0 на першому пакеті train для scratch і frozen (seed=1)
    до жодного кроку оптимізатора. Екземпляри моделей після цього не використовуються.
    """
    print("\n=== ПЕРЕВІРКА 1.3: Початкова втрата L_0 (seed=1) ===")
    criterion = nn.CrossEntropyLoss()
    first_images, first_labels, _ = next(iter(train_loader))
    first_images = first_images.to(device)
    first_labels = first_labels.to(device)

    expected_uniform = math.log(len(CLASSES))
    losses: dict[str, float] = {}

    for regime in ("scratch", "frozen"):
        model, _ = build_model_and_optimizer(regime=regime, seed=1, device=device)
        set_train_mode(model, regime)
        with torch.no_grad():
            logits = model(first_images)
            loss_val = float(criterion(logits, first_labels).item())
        losses[regime] = loss_val
        diff = loss_val - expected_uniform
        print(
            f"Режим {regime:<8}: L_0 = {loss_val:.4f} "
            f"(ln(6) = {expected_uniform:.4f}, різниця = {diff:+.4f})"
        )

    return losses


def audit_parameters(device: torch.device) -> None:
    """
    Перевіряє загальну кількість параметрів, тренованих параметрів і склад груп оптимізатора
    для всіх трьох режимів (seed=1).
    """
    print("\n=== ПЕРЕВІРКА 2.1: Аудит параметрів і груп оптимізатора (seed=1) ===")
    for regime in ("scratch", "frozen", "unfreeze"):
        model, optimizer = build_model_and_optimizer(regime=regime, seed=1, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params == 11_179_590, f"Очікувалось 11179590, отримано {total_params}"
        expected_trainable = REGIMES[regime]["expected_trainable_params"]
        assert trainable_params == expected_trainable, (
            f"[{regime}] Очікувалось {expected_trainable}, отримано {trainable_params}"
        )

        # Перевірка, що жоден заморожений параметр не потрапив в оптимізатор
        opt_param_ids = set()
        groups_desc = []
        for idx, group in enumerate(optimizer.param_groups):
            g_numel = sum(p.numel() for p in group["params"])
            for p in group["params"]:
                assert p.requires_grad, f"[{regime}] В оптимізатор потрапив заморожений параметр!"
                opt_param_ids.add(id(p))
            groups_desc.append(f"група {idx + 1}: {g_numel:,} пар. (lr={group['lr']})")

        assert len(opt_param_ids) == sum(1 for p in model.parameters() if p.requires_grad)

        print(
            f"Режим {regime:<8}: усього = {total_params:,} | "
            f"навчається = {trainable_params:,} | " + "; ".join(groups_desc)
        )


def _run_5_steps_and_diff(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    use_set_train_mode: bool,
    regime: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Виконує 5 кроків навчання і повертає state_dict до і після."""
    before_state = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()

    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()

    loader_iter = iter(train_loader)
    for _ in range(5):
        images, labels, _ = next(loader_iter)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    after_state = copy.deepcopy(model.state_dict())
    return before_state, after_state


def _inspect_module_changes(
    before_state: dict[str, torch.Tensor],
    after_state: dict[str, torch.Tensor],
    eval_modules: list[str],
) -> None:
    """Друкує таблицю змін параметрів і буферів BN по модулях верхнього рівня."""
    header = (
        f"{'Модуль':<10} | {'Параметри змінилися':<20} | "
        f"{'Ковзні статистики BN':<22} | {'Збігається з очікуваним':<24}"
    )
    print(header)
    print("-" * len(header))

    for mod_name in TOP_LEVEL_MODULES:
        prefix = f"{mod_name}."
        keys = [k for k in before_state.keys() if k.startswith(prefix)]

        bn_keys = [k for k in keys if k.endswith(BN_BUFFER_SUFFIXES)]
        param_keys = [k for k in keys if k not in bn_keys]

        params_changed = any(not torch.equal(before_state[k], after_state[k]) for k in param_keys)
        if not bn_keys:
            bn_status = "немає BN"
            bn_changed = False
        else:
            bn_changed = any(not torch.equal(before_state[k], after_state[k]) for k in bn_keys)
            bn_status = "так" if bn_changed else "ні"

        params_status = "так" if params_changed else "ні"

        is_frozen = mod_name in eval_modules
        if is_frozen:
            matches_expected = (not params_changed) and (not bn_changed)
        else:
            expected_bn = len(bn_keys) > 0
            matches_expected = params_changed and (bn_changed == expected_bn)

        print(
            f"{mod_name:<10} | {params_status:<20} | "
            f"{bn_status:<22} | {'так' if matches_expected else 'НІ':<24}"
        )


def verify_regimes_and_bn_bug(train_loader: DataLoader, device: torch.device) -> None:
    """
    Виконує повну перевірку пункту 2:
    - аудит параметрів;
    - 5 кроків із set_train_mode для frozen та unfreeze;
    - дослід із навмисною помилкою (model.train() замість set_train_mode у frozen).
    """
    audit_parameters(device)

    print("\n=== ПЕРЕВІРКА 2.2: Зміни після 5 кроків із set_train_mode (seed=1) ===")
    for regime in ("frozen", "unfreeze"):
        print(f"\n--- Режим: {regime} ---")
        model, optimizer = build_model_and_optimizer(regime=regime, seed=1, device=device)
        before, after = _run_5_steps_and_diff(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            use_set_train_mode=True,
            regime=regime,
        )
        _inspect_module_changes(before, after, REGIMES[regime]["eval_modules"])

    print("\n=== ПЕРЕВІРКА 2.3: Дослід із навмисною помилкою (frozen + model.train(), seed=1) ===")
    model_bug, optimizer_bug = build_model_and_optimizer(regime="frozen", seed=1, device=device)
    before_bug, after_bug = _run_5_steps_and_diff(
        model=model_bug,
        optimizer=optimizer_bug,
        train_loader=train_loader,
        device=device,
        use_set_train_mode=False,
        regime="frozen",
    )
    _inspect_module_changes(before_bug, after_bug, REGIMES["frozen"]["eval_modules"])

    rm_before = before_bug["layer1.0.bn1.running_mean"]
    rm_after = after_bug["layer1.0.bn1.running_mean"]
    rv_before = before_bug["layer1.0.bn1.running_var"]
    rv_after = after_bug["layer1.0.bn1.running_var"]

    max_diff_mean = float(torch.max(torch.abs(rm_after - rm_before)).item())
    max_diff_var = float(torch.max(torch.abs(rv_after - rv_before)).item())
    print(
        f"\nМаксимальна абсолютна зміна у layer1.0.bn1 після 5 кроків:\n"
        f"  running_mean: {max_diff_mean:.6f}\n"
        f"  running_var:  {max_diff_var:.6f}"
    )