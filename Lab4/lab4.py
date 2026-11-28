import argparse
import csv
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models, transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


class ImageDataReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], tfm: transforms.Compose) -> None:
        self._iter = iter(paths)
        self._tfm = tfm

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._iter, None)
        if path is None:
            return None
        with Image.open(path) as img:
            tensor = self._tfm(img.convert("RGB")).unsqueeze(0)
        return {"input": tensor.numpy().astype(np.float32)}


def load_model(path: Path) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert list(ckpt["classes"]) == CLASSES, f"Неочікувані класи: {ckpt['classes']}"

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, int(ckpt["img_size"])


def load_splits(split_path: Path, data_dir: Path) -> tuple[list[dict], list[dict]]:
    with split_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train = [r for r in rows if r["part"] == "train"]
    test = [r for r in rows if r["part"] == "test"]

    assert len(train) == 1200 and sum(int(Path(r["file"]).stem) for r in train) == 11_993_084
    assert len(test) == 1502 and sum(int(Path(r["file"]).stem) for r in test) == 33_351_444

    for r in train:
        r["path"] = data_dir / "seg_train" / r["label"] / r["file"]
    for r in test:
        r["path"] = data_dir / "seg_test" / r["label"] / r["file"]

    return train, test


def select_calibration(train_rows: list[dict]) -> list[Path]:
    sorted_train = sorted(train_rows, key=lambda r: r["file"])
    idx = np.random.default_rng(0).permutation(len(sorted_train))[:200]
    calib = [sorted_train[i] for i in idx]

    assert len(calib) == 200
    assert sum(int(Path(r["file"]).stem) for r in calib) == 1_921_470
    expected_counts = {
        "buildings": 40,
        "forest": 35,
        "glacier": 36,
        "mountain": 33,
        "sea": 29,
        "street": 27,
    }
    assert dict(Counter(r["label"] for r in calib)) == expected_counts

    return [r["path"] for r in calib]


def make_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def export_and_quantize(
    model: torch.nn.Module,
    img_size: int,
    calib_paths: list[Path],
    tfm: transforms.Compose,
    onnx_dir: Path,
) -> tuple[dict[str, Path], dict[str, float]]:
    onnx_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fp32": onnx_dir / "lab4_fp32.onnx",
        "prep": onnx_dir / "lab4_prep.onnx",
        "int8_pc": onnx_dir / "lab4_int8.onnx",
        "int8_pt": onnx_dir / "lab4_int8_per_tensor.onnx",
    }
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        torch.randn(1, 3, img_size, img_size),
        str(paths["fp32"]),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    timings["export_fp32_s"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    quant_pre_process(str(paths["fp32"]), str(paths["prep"]), skip_symbolic_shape=True)
    timings["pre_process_s"] = round(time.perf_counter() - t0, 4)

    for key, per_channel in [("int8_pc", True), ("int8_pt", False)]:
        t0 = time.perf_counter()
        quantize_static(
            model_input=str(paths["prep"]),
            model_output=str(paths[key]),
            calibration_data_reader=ImageDataReader(calib_paths, tfm),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method=CalibrationMethod.MinMax,
        )
        timings[f"quantize_{key}_s"] = round(time.perf_counter() - t0, 4)

    return paths, timings


def analyze_graph_and_weights(fp32_path: Path) -> dict:
    graph = onnx.load(str(fp32_path)).graph
    op_counts = Counter(node.op_type for node in graph.node)
    assert op_counts["Conv"] == 20 and op_counts["BatchNormalization"] == 0

    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    conv_stats = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        weights = initializers[node.input[1]]  # (C_out, C_in, kH, kW)
        alpha = np.max(np.abs(weights), axis=(1, 2, 3))
        r = float(np.max(alpha) / np.median(alpha))
        conv_stats.append(
            {
                "name": node.name,
                "r": round(r, 4),
                "bits_cost": round(float(np.log2(r)), 4),
            }
        )

    conv_stats.sort(key=lambda x: x["r"], reverse=True)
    median_r = float(np.median([x["r"] for x in conv_stats]))
    return {
        "conv_nodes": op_counts["Conv"],
        "bn_nodes": op_counts["BatchNormalization"],
        "top3_conv": conv_stats[:3],
        "median_r": round(median_r, 4),
        "median_bits_cost": round(float(np.log2(median_r)), 4),
        "all_conv": conv_stats,
    }


def evaluate_all(
    pt_model: torch.nn.Module,
    sessions: dict[str, ort.InferenceSession],
    test_rows: list[dict],
    tfm: transforms.Compose,
    batch_size: int = 64,
) -> tuple[dict, np.ndarray]:
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_true = np.array([class_to_idx[r["label"]] for r in test_rows])
    logits = {k: [] for k in ["pytorch", *sessions.keys()]}
    first_sample = None

    with torch.no_grad():
        for start in range(0, len(test_rows), batch_size):
            batch_rows = test_rows[start : start + batch_size]
            imgs = []
            for r in batch_rows:
                with Image.open(r["path"]) as img:
                    imgs.append(tfm(img.convert("RGB")))
            batch_tensor = torch.stack(imgs)
            batch_np = batch_tensor.numpy().astype(np.float32)

            if first_sample is None:
                first_sample = batch_np[:1].copy()

            logits["pytorch"].append(pt_model(batch_tensor).numpy())
            for name, sess in sessions.items():
                logits[name].append(sess.run(["logits"], {"input": batch_np})[0])

    logits_np = {k: np.concatenate(v, axis=0) for k, v in logits.items()}
    preds = {k: np.argmax(v, axis=1) for k, v in logits_np.items()}

    max_diff = float(np.max(np.abs(logits_np["pytorch"] - logits_np["fp32"])))
    assert max_diff < 1e-3, f"Розбіжність логітів занадто велика: {max_diff}"

    fp32_correct = preds["fp32"] == y_true
    n = len(y_true)
    metrics = {
        "export_check": {
            "max_abs_logit_diff": max_diff,
            "class_match_rate": float(np.mean(preds["pytorch"] == preds["fp32"])),
            "pytorch_macro_f1": round(float(f1_score(y_true, preds["pytorch"], average="macro")), 4),
            "fp32_macro_f1": round(float(f1_score(y_true, preds["fp32"], average="macro")), 4),
        },
        "variants": {},
    }

    for name in ["fp32", "int8_pc", "int8_pt"]:
        var_correct = preds[name] == y_true
        b = int(np.sum(var_correct & ~fp32_correct))
        c = int(np.sum(~var_correct & fp32_correct))
        delta_acc = (b - c) / n
        se = float(np.sqrt(b + c) / n)
        metrics["variants"][name] = {
            "macro_f1": round(float(f1_score(y_true, preds[name], average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_true, preds[name])), 4),
            "agreement_with_fp32": round(float(np.mean(preds[name] == preds["fp32"])), 4),
            "b": b,
            "c": c,
            "delta_acc": round(delta_acc, 4),
            "se_delta_acc": round(se, 4),
            "significant_vs_noise": bool(abs(b - c) > 2 * np.sqrt(b + c)),
        }

    return metrics, first_sample


def measure_latency(
    sessions: dict[str, ort.InferenceSession],
    sample: np.ndarray,
    rounds: int = 3,
    warmup: int = 20,
    iters: int = 100,
) -> dict:
    raw_ms = {name: [[] for _ in range(rounds)] for name in sessions}

    for r in range(rounds):
        for name, sess in sessions.items():
            for _ in range(warmup):
                sess.run(["logits"], {"input": sample})
            for _ in range(iters):
                t0 = time.perf_counter()
                sess.run(["logits"], {"input": sample})
                raw_ms[name][r].append((time.perf_counter() - t0) * 1000.0)

    summary = {}
    for name, round_lists in raw_ms.items():
        all_runs = [x for rnd in round_lists for x in rnd]
        round_p50 = [round(float(np.percentile(rnd, 50)), 3) for rnd in round_lists]
        summary[name] = {
            "p50_ms": round(float(np.percentile(all_runs, 50)), 3),
            "p95_ms": round(float(np.percentile(all_runs, 95)), 3),
            "round_p50_ms": round_p50,
            "raw_rounds_ms": [[round(x, 4) for x in rnd] for rnd in round_lists],
        }

    fp32_r = summary["fp32"]["round_p50_ms"]
    int8_r = summary["int8"]["round_p50_ms"]
    non_overlapping = max(int8_r) < min(fp32_r) or max(fp32_r) < min(int8_r)
    summary["speedup_p50"] = round(summary["fp32"]["p50_ms"] / summary["int8"]["p50_ms"], 3)
    summary["measured_change_non_overlapping"] = bool(non_overlapping)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР4: Експорт FP32 та статична квантизація INT8")
    parser.add_argument("--model", type=Path, default=Path("../Lab3/lab3_model.pt"))
    parser.add_argument("--split", type=Path, default=Path("../Lab3/split.csv"))
    parser.add_argument("--data", type=Path, default=Path("../Lab3"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("."))
    parser.add_argument("--results", type=Path, default=Path("lab4_results.json"))
    args = parser.parse_args()

    pt_model, img_size = load_model(args.model)
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_rows, test_rows = load_splits(args.split, args.data)
    calib_paths = select_calibration(train_rows)

    onnx_paths, build_timings = export_and_quantize(
        pt_model, img_size, calib_paths, tfm, args.onnx_dir
    )
    graph_analysis = analyze_graph_and_weights(onnx_paths["fp32"])

    eval_sessions = {
        "fp32": make_session(onnx_paths["fp32"]),
        "int8_pc": make_session(onnx_paths["int8_pc"]),
        "int8_pt": make_session(onnx_paths["int8_pt"]),
    }
    quality_metrics, first_sample = evaluate_all(pt_model, eval_sessions, test_rows, tfm)

    latency_sessions = {
        "fp32": eval_sessions["fp32"],
        "int8": eval_sessions["int8_pc"],
    }
    latency_metrics = measure_latency(latency_sessions, first_sample)

    file_sizes_mb = {
        k: round(p.stat().st_size / 1e6, 3) for k, p in onnx_paths.items() if k != "prep"
    }

    results = {
        "environment": {
            "cpu": platform.processor() or platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "num_threads": 1,
            "latency_batch_size": 1,
            "eval_batch_size": 64,
        },
        "build_timings": build_timings,
        "file_sizes_mb": file_sizes_mb,
        "graph_and_weights": graph_analysis,
        "quality": quality_metrics,
        "latency": latency_metrics,
    }

    args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Готово. Результати збережено у {args.results}")


if __name__ == "__main__":
    main()