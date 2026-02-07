#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# ============================= Config =============================
CANDIDATE_KEYS = [
    "birads", "breast_density", "density",
    "mass", "calcification", "asymmetry",
    "malignancy", "status"
]

OUTPUT_CSV_NAME = "macro_metrics.csv"         # single wide CSV
MATRICES_DIR    = "matrices"                  # where confusion matrices go
ROC_DIR         = "roc_curves"                # where ROC curves go
LOGS_DIR        = "task_logs"                 # where per-task text logs go

# If your pred JSONs contain probabilities for a binary task, add the probable keys here.
PROB_KEYS = ["prob", "score", "probability"]

# ============================= Utils =============================
def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('_')

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ===================== Normalizers by task/type ====================
def norm_birads(v) -> str | None:
    """Return string '0'..'6' if possible (or '1'..'5'), else None."""
    if v is None: return None
    s = str(v).strip()
    m = re.search(r"(?:^|[^0-9])([0-6])(?:[^0-9]|$)", s)
    return m.group(1) if m else None

def norm_density(v) -> str | None:
    """Return 'A'/'B'/'C'/'D' if possible, else None."""
    if v is None: return None
    s = str(v).strip()
    m = re.search(r"\b([ABCD])\b", s, flags=re.IGNORECASE)
    if m: return m.group(1).upper()
    # fallback: last char if it looks like density
    last = s[-1:].upper()
    return last if last in {"A", "B", "C", "D"} else None

def norm_yesno(v) -> str | None:
    """Return 'yes' or 'no' from strings/bools/0-1; else None."""
    if v is None: return None
    s = str(v).strip().lower()
    if s in {"yes", "y", "true", "1"}: return "yes"
    if s in {"no", "n", "false", "0"}: return "no"
    return None

def norm_malignancy(v) -> str | None:
    """Return 'healthy'/'benign'/'malignant' if possible, else None."""
    if v is None: return None
    s = str(v).strip().lower()
    if s in {"healthy", "benign", "malignant"}: return s
    if "healthy" in s: return "healthy"
    if "benign" in s: return "benign"
    if "malignan" in s or "cancer" in s: return "malignant"
    return None

def birads_to_malignancy(birads_value) -> str | None:
    """
    Map a BIRADS value to malignancy class:
      1 -> healthy
      2,3 -> benign
      4,5 -> malignant
    Anything else (0,6,unknown) -> None (ignored).
    """
    code = norm_birads(birads_value)  # returns '0'..'6' or None
    if code is None:
        return None
    if code == "1":
        return "healthy"
    if code in {"2", "3"}:
        return "benign"
    if code in {"4", "5"}:
        return "malignant"
    # e.g., BIRADS 0 or 6 → skip
    return None

def choose_task_key(d: Dict[str, Any]) -> str | None:
    """Pick which key from the prediction JSON to evaluate."""
    # Prefer known keys in order
    for k in CANDIDATE_KEYS:
        if k in d:
            return k
    # Otherwise, pick the first non-structural, non-path key with scalar/short string
    for k, v in d.items():
        if k in {"image", "image_id", "path"}:
            continue
        if isinstance(v, (str, int, float, bool)):
            return k
    return None

def normalize_value(task_key: str, value: Any) -> str | None:
    """Normalize (task_key, value) to a comparable class label string."""
    if task_key == "birads":
        return norm_birads(value)
    if task_key in {"breast_density", "density"}:
        return norm_density(value)
    if task_key in {"mass", "calcification", "asymmetry", "status"}:
        return norm_yesno(value)
    if task_key == "malignancy":
        return norm_malignancy(value)
    # Default: string compare
    return str(value).strip() if value is not None else None

def is_binary_task(task_key: str) -> bool:
    return task_key in {"mass", "calcification", "asymmetry", "status"}

def extract_prob(pred_obj: Dict[str, Any], task_key: str) -> float | None:
    """
    Try to find a probability for ROC/AUC in binary tasks.
    Looks for <task>_prob or generic keys like 'prob', 'score', 'probability'.
    Must be a float in [0,1].
    """
    # prefer task-specific key like "mass_prob"
    tk_prob = f"{task_key}_prob"
    if tk_prob in pred_obj:
        try:
            p = float(pred_obj[tk_prob])
            if 0.0 <= p <= 1.0:
                return p
        except Exception:
            pass
    for k in PROB_KEYS:
        if k in pred_obj:
            try:
                p = float(pred_obj[k])
                if 0.0 <= p <= 1.0:
                    return p
            except Exception:
                pass
    return None

# ============= NEW: extract GT from 'findings' for some tasks =============
def extract_binary_from_findings(task_key: str, gt_obj: Dict[str, Any]) -> str | None:
    """
    For calcification, mass, and asymmetry, derive GT from the 'findings' text.
    Presence of the corresponding word -> 'yes', absence -> 'no'.
    """
    findings = gt_obj.get("findings", "")
    if findings is None:
        findings = ""
    s = str(findings).lower()

    if task_key == "calcification":
        # calcification / calcifications
        if re.search(r"\bcalcifications?\b", s):
            return "yes"
        else:
            return "no"
    elif task_key == "mass":
        # mass / masses
        if re.search(r"\bmass(es)?\b", s):
            return "yes"
        else:
            return "no"
    elif task_key == "asymmetry":
        # asymmetry / asymmetries / asymmetric / asymmetrically, etc.
        if re.search(r"\basymmet\w*\b", s):
            return "yes"
        else:
            return "no"
    return None
# ===================================================================

# ===================== Metrics & plots ============================
def macro_specificity(y_true, y_pred, labels=None) -> float:
    if not y_true: return 0.0
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specs))

def macro_block(y_true, y_pred, labels=None) -> Dict[str, float]:
    if not y_true:
        return {
            "Macro Accuracy":    0.0,
            "Macro Precision":   0.0,
            "Macro Recall":      0.0,
            "Macro F1":          0.0,
            "Macro Specificity": 0.0
        }
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    # Macro accuracy = mean(per-class accuracy over true labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_acc = []
    for i in range(len(labels)):
        tot_i = cm[i, :].sum()
        acc_i = (cm[i, i] / tot_i) if tot_i else 0.0
        per_class_acc.append(acc_i)
    macro_accuracy = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    macro_precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_recall    = recall_score   (y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_f1        = f1_score       (y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_spec      = macro_specificity(y_true, y_pred, labels)
    return {
        "Macro Accuracy":    macro_accuracy,
        "Macro Precision":   float(macro_precision),
        "Macro Recall":      float(macro_recall),
        "Macro F1":          float(macro_f1),
        "Macro Specificity": float(macro_spec),
    }

def save_confusion_png(y_true, y_pred, labels, title, out_path: Path):
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted label',
        ylabel='True label',
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center", fontweight='bold',
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def try_save_roc_auc(y_true, y_score, pos_label: str, title: str, out_path: Path) -> float | None:
    """
    Save ROC curve and return AUC if we have usable scores (binary only).
    y_true: list of gold labels ("yes"/"no" e.g.)
    y_score: list of predicted probabilities/proportions for the positive class (float in [0,1]).
    """
    if not y_true or not y_score or len(y_true) != len(y_score):
        return None
    # Build binary ground-truth
    y_true_bin = np.array([1 if t == pos_label else 0 for t in y_true], dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    # If all positives or all negatives, ROC is undefined
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return None

    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    AUC = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {AUC:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return float(AUC)

# ===================== Main evaluator =============================
def evaluate(gt_dir: str, pred_root: str):
    gt_dir = Path(gt_dir).resolve()
    pred_root = Path(pred_root).resolve()

    out_csv = (pred_root / OUTPUT_CSV_NAME)
    matrices_dir = ensure_dir(pred_root / MATRICES_DIR)
    roc_dir      = ensure_dir(pred_root / ROC_DIR)
    logs_dir     = ensure_dir(pred_root / LOGS_DIR)

    # Find leaf prediction directories that contain JSONs
    leaf_dirs: List[Path] = []
    for p in pred_root.rglob("*"):
        if p.is_dir():
            if any(child.suffix.lower() == ".json" for child in p.iterdir() if child.is_file()):
                leaf_dirs.append(p.resolve())

    rows = []  # rows for CSV

    if not leaf_dirs:
        print(f"No prediction subdirectories with .json found under: {pred_root}")
        return

    for pred_dir in sorted(leaf_dirs):
        prompt = pred_dir.name
        model  = pred_dir.parent.name

        # Aggregate per-task: {task_key: list of (y_true, y_pred, prob_opt)}
        task_true: Dict[str, List[str]] = {}
        task_pred: Dict[str, List[str]] = {}
        task_prob: Dict[str, List[float]] = {}  # optional, for ROC/AUC (binary)
        per_task_labels: Dict[str, set] = {}    # record all labels seen

        # Special aggregation for malignancy using GT BIRADS
        mal_true: List[str] = []
        mal_pred: List[str] = []

        # Iterate over prediction JSONs
        for f in pred_dir.iterdir():
            if not (f.is_file() and f.suffix.lower() == ".json"):
                continue
            fname = f.name
            gt_path = gt_dir / fname
            if not gt_path.exists():
                # no matching GT → skip this file
                continue

            try:
                pr = read_json(f)
                gt = read_json(gt_path)
            except Exception:
                continue

            # -------- Special malignancy evaluation (pred: "malignancy", GT: "birads") --------
            if "malignancy" in pr and "birads" in gt:
                y_p_mal = norm_malignancy(pr.get("malignancy"))
                y_t_mal = birads_to_malignancy(gt.get("birads"))
                if y_p_mal is not None and y_t_mal is not None:
                    mal_true.append(y_t_mal)
                    mal_pred.append(y_p_mal)
            # -------------------------------------------------------------------------------

            task_key = choose_task_key(pr)
            if not task_key:
                continue

            y_p = normalize_value(task_key, pr.get(task_key))

            # ==== NEW: take GT for calcification/mass/asymmetry from 'findings' ====
            if task_key in {"calcification", "mass", "asymmetry"}:
                y_t = extract_binary_from_findings(task_key, gt)
            else:
                y_t = normalize_value(task_key, gt.get(task_key))
            # =======================================================================

            if y_p is None or y_t is None:
                continue

            task_true.setdefault(task_key, []).append(y_t)
            task_pred.setdefault(task_key, []).append(y_p)
            per_task_labels.setdefault(task_key, set()).update([y_t, y_p])

            # Optional probability for ROC/AUC (only for binary tasks)
            if is_binary_task(task_key):
                pscore = extract_prob(pr, task_key)
                if pscore is not None:
                    task_prob.setdefault(task_key, []).append(pscore)
                else:
                    # keep alignment if any prob already stored
                    if task_key in task_prob:
                        task_prob[task_key].append(None)

        # For each task gathered in this (model, prompt) dir, compute outputs
        for task_key in sorted(task_true.keys()):
            y_true = task_true[task_key]
            y_pred = task_pred[task_key]
            labels = sorted(per_task_labels.get(task_key, set()))

            # Macro metrics
            block = macro_block(y_true, y_pred, labels=labels)

            # Confusion matrix
            cm_title = f"{task_key} ({model}/{prompt})"
            cm_path  = matrices_dir / f"confusion__{_slug(task_key)}__{_slug(model)}__{_slug(prompt)}.png"
            try:
                save_confusion_png(y_true, y_pred, labels, cm_title, cm_path)
            except Exception:
                pass

            # ROC/AUC (only for binary AND if probabilities available)
            auc_value = ""
            if is_binary_task(task_key) and task_key in task_prob:
                # Filter pairs where we have a numeric prob
                probs = task_prob[task_key]
                pairs = [(t, p, s) for t, p, s in zip(y_true, y_pred, probs) if isinstance(s, (int, float))]
                if pairs:
                    y_true_filt = [t for (t, _, _) in pairs]
                    y_score_filt = [float(s) for (_, _, s) in pairs]
                    # Positive class: pick the "yes" for yes/no, otherwise the last class (rare case)
                    pos_label = "yes" if set(labels) == {"no", "yes"} else sorted(labels)[-1]
                    roc_title = f"ROC {task_key} ({model}/{prompt})"
                    roc_path  = roc_dir / f"roc__{_slug(task_key)}__{_slug(model)}__{_slug(prompt)}.png"
                    try:
                        AUC = try_save_roc_auc(y_true_filt, y_score_filt, pos_label, roc_title, roc_path)
                        if AUC is not None:
                            auc_value = f"{AUC:.4f}"
                    except Exception:
                        pass

            # Write per-task TXT log with per-class metrics
            log = []
            log.append(f"=== {task_key} | model={model} | prompt={prompt}")
            log.append(f"Samples: {len(y_true)}")
            log.append(f"Labels: {labels}\n")

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            log.append("Confusion matrix (rows=true, cols=pred):")
            for i, lab in enumerate(labels):
                row = "  ".join(f"{cm[i, j]:4d}" for j in range(len(labels)))
                log.append(f"{lab:>10s} | {row}")
            log.append("")

            # per-class metrics
            log.append("Per-class metrics:")
            for i, lab in enumerate(labels):
                TP = cm[i, i]
                FP = cm[:, i].sum() - TP
                FN = cm[i, :].sum() - TP
                TN = cm.sum() - TP - FP - FN
                prec = TP / (TP + FP) if (TP + FP) else 0.0
                rec  = TP / (TP + FN) if (TP + FN) else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                spec = TN / (TN + FP) if (TN + FP) else 0.0
                log.append(f"- {lab:>10s} | P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  Spec={spec:.4f}")

            # summary (macro)
            log.append("")
            log.append("Macro metrics:")
            log.append(f"- Macro Accuracy    : {block['Macro Accuracy']:.4f}")
            log.append(f"- Macro Precision   : {block['Macro Precision']:.4f}")
            log.append(f"- Macro Recall      : {block['Macro Recall']:.4f}")
            log.append(f"- Macro F1          : {block['Macro F1']:.4f}")
            log.append(f"- Macro Specificity : {block['Macro Specificity']:.4f}")
            if auc_value != "":
                log.append(f"- AUC               : {auc_value}")

            task_log_path = (Path(logs_dir) / f"results__{_slug(task_key)}__{_slug(model)}__{_slug(prompt)}.txt")
            with task_log_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(log) + "\n")

            # Add CSV row
            rows.append({
                "Model": model,
                "Prompt": prompt,
                "Task": task_key,
                "Macro Accuracy":    f"{block['Macro Accuracy']:.4f}",
                "Macro Precision":   f"{block['Macro Precision']:.4f}",
                "Macro Recall":      f"{block['Macro Recall']:.4f}",
                "Macro F1":          f"{block['Macro F1']:.4f}",
                "Macro Specificity": f"{block['Macro Specificity']:.4f}",
                "AUC": auc_value
            })

        # === Special malignancy metrics (pred "malignancy" vs GT(BIRADS→malignancy)) ===
        if mal_true:
            labels = sorted(set(mal_true) | set(mal_pred))
            block = macro_block(mal_true, mal_pred, labels=labels)

            # Confusion matrix
            cm_title = f"malignancy (from BIRADS) ({model}/{prompt})"
            cm_path  = matrices_dir / f"confusion__malignancy_from_birads__{_slug(model)}__{_slug(prompt)}.png"
            try:
                save_confusion_png(mal_true, mal_pred, labels, cm_title, cm_path)
            except Exception:
                pass

            # Per-task TXT log for malignancy
            log = []
            log.append(f"=== malignancy (from BIRADS) | model={model} | prompt={prompt}")
            log.append("Mapping: BIRADS 1→healthy, 2–3→benign, 4–5→malignant")
            log.append(f"Samples: {len(mal_true)}")
            log.append(f"Labels: {labels}\n")

            cm = confusion_matrix(mal_true, mal_pred, labels=labels)
            log.append("Confusion matrix (rows=true, cols=pred):")
            for i, lab in enumerate(labels):
                row = "  ".join(f"{cm[i, j]:4d}" for j in range(len(labels)))
                log.append(f"{lab:>10s} | {row}")
            log.append("")

            # Per-class metrics
            log.append("Per-class metrics:")
            for i, lab in enumerate(labels):
                TP = cm[i, i]
                FP = cm[:, i].sum() - TP
                FN = cm[i, :].sum() - TP
                TN = cm.sum() - TP - FP - FN
                prec = TP / (TP + FP) if (TP + FP) else 0.0
                rec  = TP / (TP + FN) if (TP + FN) else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                log.append(f"- {lab:>10s} | P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

            # Summary (macro)
            log.append("")
            log.append("Macro metrics:")
            log.append(f"- Accuracy (macro)  : {block['Macro Accuracy']:.4f}")
            log.append(f"- Precision (macro) : {block['Macro Precision']:.4f}")
            log.append(f"- Recall (macro)    : {block['Macro Recall']:.4f}")
            log.append(f"- F1 (macro)        : {block['Macro F1']:.4f}")

            task_log_path = (Path(logs_dir) /
                             f"results__malignancy_from_birads__{_slug(model)}__{_slug(prompt)}.txt")
            with task_log_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(log) + "\n")

            # Add row to CSV
            rows.append({
                "Model": model,
                "Prompt": prompt,
                "Task": "malignancy_from_birads",
                "Macro Accuracy":    f"{block['Macro Accuracy']:.4f}",
                "Macro Precision":   f"{block['Macro Precision']:.4f}",
                "Macro Recall":      f"{block['Macro Recall']:.4f}",
                "Macro F1":          f"{block['Macro F1']:.4f}",
                "Macro Specificity": f"{block['Macro Specificity']:.4f}",
                "AUC": ""  # multi-class malignancy → no AUC here
            })
        # === End malignancy metrics ===

        print(f"✅ Finished: model='{model}', prompt='{prompt}'")

    # Write one CSV aggregating all (model, prompt, task)
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=[
            "Model", "Prompt", "Task",
            "Macro Accuracy", "Macro Precision", "Macro Recall",
            "Macro F1", "Macro Specificity", "AUC"
        ])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"📄 Wrote CSV: {out_csv}")
    print(f"🖼  Confusions: {matrices_dir}")
    print(f"📈 ROC curves: {roc_dir}")
    print(f"📝 Logs:       {logs_dir}")

# =========================== Entrypoint ===========================
if __name__ == "__main__":
    gt_dir   = "..."
    pred_dir = "..."
    evaluate(gt_dir, pred_dir)
