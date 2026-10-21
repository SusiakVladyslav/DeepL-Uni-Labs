import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

from lab2_starter import (
    make_splits,
    train_once,
    save_run,
    load_predictions,
    split_summary
)


def fast_threshold_search(scores: np.ndarray, labels: np.ndarray):
    n = len(scores)
    total_pos = np.sum(labels == 1)
    total_neg = n - total_pos

    order = np.argsort(-scores)
    s_sorted = scores[order]
    y_sorted = labels[order]

    distinct_mask = np.empty(n, dtype=bool)
    distinct_mask[:-1] = s_sorted[:-1] != s_sorted[1:]
    distinct_mask[-1] = True
    distinct_indices = np.where(distinct_mask)[0]

    cum_tp = np.cumsum(y_sorted)[distinct_indices]
    cum_fp = (distinct_indices + 1) - cum_tp

    thresholds = s_sorted[distinct_indices]
    tp = cum_tp
    fp = cum_fp

    thresholds = np.append(thresholds, np.inf)
    tp = np.append(tp, 0)
    fp = np.append(fp, 0)
    fn = total_pos - tp
    tn = total_neg - fp

    costs = 10 * fn + fp

    min_cost = np.min(costs)
    candidate_indices = np.where(costs == min_cost)[0]
    best_idx = candidate_indices[np.argmax(thresholds[candidate_indices])]

    best_t = thresholds[best_idx]

    candidates_df = pd.DataFrame({
        'threshold': thresholds,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'C': costs
    })

    return best_t, candidates_df


def slow_threshold_search(scores: np.ndarray, labels: np.ndarray):
    unique_s = np.unique(scores)
    candidates = np.sort(np.append(unique_s, np.inf))[::-1]

    records = []
    for t in candidates:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        c = 10 * fn + fp
        records.append({'threshold': t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'C': c})

    df = pd.DataFrame(records)
    min_c = df['C'].min()
    best_t = df[df['C'] == min_c]['threshold'].max()

    return best_t, df

def run_step3_validation_and_unit_tests(val_scores, y_val, out_dir: Path):
    print("\n==========================================")
    print("КРОК 3.1. МОДУЛЬНЕ ТЕСТУВАННЯ (Unit-тести)")
    print("==========================================")

    test_sets = {
        'A (Рівні оцінки)': (
            np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1]),
            np.array([0, 1, 0, 1, 0, 0])
        ),
        'B (Без позитивних міток)': (
            np.array([0.9, 0.6, 0.3]),
            np.array([0, 0, 0])
        ),
        'C (Однакова вартість на кількох порогах)': (
            np.array([0.8] + [0.5] * 11 + [0.2]),
            np.array([1, 1] + [0] * 10 + [0])
        )
    }

    all_passed = True
    for name, (s_test, y_test) in test_sets.items():
        t_fast, df_fast = fast_threshold_search(s_test, y_test)
        t_slow, df_slow = slow_threshold_search(s_test, y_test)

        same_t = np.isclose(t_fast, t_slow) or (np.isinf(t_fast) and np.isinf(t_slow))
        df_fast_sorted = df_fast.sort_values('threshold', ascending=False).reset_index(drop=True)
        df_slow_sorted = df_slow.sort_values('threshold', ascending=False).reset_index(drop=True)

        same_tables = np.allclose(df_fast_sorted[['TP', 'FP', 'FN', 'TN', 'C']],
                                  df_slow_sorted[['TP', 'FP', 'FN', 'TN', 'C']])

        print(f"\n--- Таблиця перевірки для Набору {name} ---")
        df_display = df_fast_sorted.copy()
        df_display['threshold'] = df_display['threshold'].apply(lambda x: '+inf' if np.isinf(x) else f"{x:.1f}")
        print(df_display.to_string(index=False))

        if same_t and same_tables:
            print(f"✓ Набір {name}: ТЕСТ ПУЩЕНО (Обраний поріг t* = {t_fast})")
        else:
            print(f"✗ Набір {name}: ТЕСТ ПРОВАЛЕНО")
            all_passed = False

    assert all_passed, "Тести не пройдені! Перевірте реалізацію алгоритму."

    print("\n==========================================")
    print("КРОК 3.2. ПОШУК ПОРОГА НА VALIDATION")
    print("==========================================")

    t_star, candidates_df = fast_threshold_search(val_scores, y_val)
    candidates_df.to_csv(out_dir / 'validation_candidates.csv', index=False)

    fpr, tpr, _ = roc_curve(y_val, val_scores)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_scores)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Validation)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, color='orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Validation)')
    plt.tight_layout()
    plt.savefig(out_dir / 'curves.png')
    plt.close()

    print(f"ROC-AUC на Validation: {roc_auc:.4f}")
    print(f"Оптимальний поріг t*: {t_star}")

    rules = {
        'Standard (t=0.5)': 0.5,
        'Optimal (t*)': t_star,
        'All Negative (inf)': np.inf
    }

    val_summary = []
    for name, t_val in rules.items():
        preds = (val_scores >= t_val).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        tn = np.sum((preds == 0) & (y_val == 0))
        c = 10 * fn + fp
        val_summary.append({'Правило': name, 'Поріг': t_val, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'C': c})

    print("\nПідсумки на Validation:")
    print(pd.DataFrame(val_summary).to_string(index=False))

    return t_star



def run_step4_test_evaluation(test_scores, y_test, idx_test, test_amount, t_star):
    print("\n==========================================")
    print("КРОК 4. ТЕСТОВА ОЦІНКА ТА АНАЛІЗ ПОМИЛОК")
    print("==========================================")

    test_preds = (test_scores >= t_star).astype(int)

    tp = np.sum((test_preds == 1) & (y_test == 1))
    fp = np.sum((test_preds == 1) & (y_test == 0))
    fn = np.sum((test_preds == 0) & (y_test == 1))
    tn = np.sum((test_preds == 0) & (y_test == 0))
    c_test = 10 * fn + fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    print(f"Матриця плутанини на Test (поріг t* = {t_star:.6f}):")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Загальна вартість C: {c_test}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")

    fp_indices = np.where((test_preds == 1) & (y_test == 0))[0][:5]
    fn_indices = np.where((test_preds == 0) & (y_test == 1))[0][:5]

    errors_records = []

    for idx in fp_indices:
        errors_records.append({
            'Рядок CSV': idx_test[idx],
            'Тип помилки': 'FP',
            'Мітка y': 0,
            'Оцінка s_i': test_scores[idx],
            '|s_i - t*|': abs(test_scores[idx] - t_star),
            'Amount': test_amount[idx]
        })

    for idx in fn_indices:
        errors_records.append({
            'Рядок CSV': idx_test[idx],
            'Тип помилки': 'FN',
            'Мітка y': 1,
            'Оцінка s_i': test_scores[idx],
            '|s_i - t*|': abs(test_scores[idx] - t_star),
            'Amount': test_amount[idx]
        })

    errors_df = pd.DataFrame(errors_records)
    print("\nПерші 5 FP та 5 FN помилок на тесті:")
    print(errors_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(
        description="ЛР2: Підготовка даних, навчання, пошук порога та тестова оцінка."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data", type=Path, help="Шлях до creditcard.csv для навчання")
    mode.add_argument("--saved", type=Path, help="Тека з результатами для аналізу без навчання")
    parser.add_argument("--out", type=Path, help="Тека для результатів навчання")
    args = parser.parse_args()

    if args.saved is not None:
        if args.out is not None:
            parser.error("У режимі --saved параметр --out не потрібний.")

        saved_dir = args.saved
        saved_data = load_predictions(saved_dir)
        print("--- Завантажена статистика поділу ---")
        print(split_summary(saved_data).to_string(index=False))

        t_star = run_step3_validation_and_unit_tests(
            saved_data['validation_scores'],
            saved_data['validation_y'],
            saved_dir
        )

        run_step4_test_evaluation(
            saved_data['test_scores'],
            saved_data['test_y'],
            saved_data['test_row_ids'],
            saved_data['test_amount'],
            t_star
        )
        return

    if args.out is None:
        parser.error("Для навчання вкажіть --out.")

    out_dir = args.out
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        parser.error("Тека --out непорожня. Для аналізу скористайтеся параметром --saved.")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_split, val_split, test_split, scaler = make_splits(args.data)
    model = train_once(train_split)
    saved_data = save_run(out_dir, model, (train_split, val_split, test_split), scaler, args.data)

    summary = split_summary(saved_data)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    print("\n--- Статистика поділу даних ---")
    print(summary.to_string(index=False))

    t_star = run_step3_validation_and_unit_tests(
        saved_data['validation_scores'],
        saved_data['validation_y'],
        out_dir
    )

    run_step4_test_evaluation(
        saved_data['test_scores'],
        saved_data['test_y'],
        saved_data['test_row_ids'],
        saved_data['test_amount'],
        t_star
    )


if __name__ == "__main__":
    main()