import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from Lab3.config import CLASSES, REGIMES, WEIGHT_DECAY


def get_device() -> torch.device:
    """Повертає найкращий доступний пристрій для навчання (CUDA, MPS або CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_and_optimizer(
    regime: str,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """
    Будує модель ResNet-18 та оптимізатор AdamW у фіксованому порядку:
    1. Фіксація зерна генераторів випадкових чисел.
    2. Ініціалізація resnet18.
    3. Заміна голови model.fc на Linear(512, 6).
    4. Налаштування requires_grad відповідно до режиму.
    5. Перенесення на пристрій та створення AdamW.
    """
    if regime not in REGIMES:
        raise ValueError(f"Невідомий режим навчання: {regime}")

    cfg = REGIMES[regime]

    # 1. Фіксація генераторів випадкових чисел
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Ініціалізація моделі
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    model = resnet18(weights=weights)

    # 3. Заміна класифікаційної голови
    model.fc = nn.Linear(512, len(CLASSES))

    # 4. Заморожування всіх параметрів і розморожування лише цільових модулів
    for param in model.parameters():
        param.requires_grad = False

    for module_name in cfg["trainable_modules"]:
        module = getattr(model, module_name)
        for param in module.parameters():
            param.requires_grad = True

    # 5. Перенесення на пристрій і створення груп оптимізатора
    model = model.to(device)

    param_groups = []
    for group_cfg in cfg["lr_groups"]:
        group_params = []
        for module_name in group_cfg["modules"]:
            module = getattr(model, module_name)
            group_params.extend(p for p in module.parameters() if p.requires_grad)
        param_groups.append({"params": group_params, "lr": group_cfg["lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def set_train_mode(model: nn.Module, regime: str) -> None:
    """
    Переводить модель у режим навчання, але примусово перемикає
    заморожені модулі кістяка в режим eval() (захист ковзних статистик BN).
    """
    if regime not in REGIMES:
        raise ValueError(f"Невідомий режим навчання: {regime}")

    model.train()
    for module_name in REGIMES[regime]["eval_modules"]:
        module = getattr(model, module_name)
        module.eval()