"""Evaluate the packaged OrganCNN weights on the OrganMNIST3D test
split and produce the failure analysis + packaged sample volumes.

Outputs (defaults):
  assets/evaluation_results.json  — full test metrics, per-class
                                    results, confusion-pair analysis,
                                    and the complete error inventory
  assets/samples/*.npy            — sample test volumes (uint8 28^3)
  assets/samples/index.json       — per-sample label + the prediction
                                    measured from the packaged .npy
                                    file itself (via organcnn.infer)

Sample selection rule (fixed, stated here so it cannot be mistaken
for cherry-picking): from the test split, in test order unless noted —
  * the first misclassified volume,
  * the most confident misclassified volume,
  * the most confident correct volume,
  * the median correct volume in test order.
If the model made no errors, the two error roles are absent and the
index records the rule; if it made no correct predictions, the two
correct roles are absent.

Usage:
  python scripts/evaluate.py [--weights model.safetensors]
      [--data /path/to/organmnist3d.npz] [--out assets/evaluation_results.json]
      [--samples-dir assets/samples]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from organcnn.infer import LABELS, TRAIN_MEAN, TRAIN_STD, load_model, predict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
NUM_CLASSES = 11


@torch.no_grad()
def evaluate(model, images_u8, batch=64):
    model.eval()
    xt = torch.from_numpy(images_u8)[:, None]
    probs = []
    for i in range(0, len(xt), batch):
        xb = (xt[i:i + batch].float() / 255.0 - TRAIN_MEAN) / TRAIN_STD
        probs.append(torch.softmax(model(xb), dim=1))
    probs = torch.cat(probs).numpy()
    return probs, probs.argmax(1)


def full_metrics(y_true, probs, preds):
    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))
    rec = recall_score(y_true, preds, labels=list(range(NUM_CLASSES)),
                       average=None, zero_division=0)
    prec = precision_score(y_true, preds, labels=list(range(NUM_CLASSES)),
                           average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "auc_macro_ovr": round(float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": {LABELS[i]: round(float(rec[i]), 4) for i in range(NUM_CLASSES)},
        "per_class_precision": {LABELS[i]: round(float(prec[i]), 4) for i in range(NUM_CLASSES)},
        "class_counts": np.bincount(y_true, minlength=NUM_CLASSES).tolist(),
        "majority_baseline": round(float(np.bincount(y_true).max() / len(y_true)), 4),
        "n": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(PKG, "model.safetensors"))
    ap.add_argument("--data", required=True, help="path to organmnist3d.npz")
    ap.add_argument("--out", default=os.path.join(PKG, "assets", "evaluation_results.json"))
    ap.add_argument("--samples-dir", default=os.path.join(PKG, "assets", "samples"))
    args = ap.parse_args()

    d = np.load(args.data)
    x_test = d["test_images"]
    y_test = d["test_labels"].ravel().astype(np.int64)

    model = load_model(args.weights)
    probs, preds = evaluate(model, x_test)
    metrics = full_metrics(y_test, probs, preds)

    correct = preds == y_test
    occupancy = (x_test > 0).sum(axis=(1, 2, 3))
    conf_pred = probs[np.arange(len(y_test)), preds]
    conf_true = probs[np.arange(len(y_test)), y_test]

    # Confusion pairs: every off-diagonal (true -> predicted) cell.
    cm = np.array(metrics["confusion_matrix"])
    pairs = []
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if t != p and cm[t, p] > 0:
                pairs.append({
                    "true": LABELS[t], "predicted": LABELS[p],
                    "count": int(cm[t, p]),
                    "share_of_true_class": round(float(cm[t, p] / cm[t].sum()), 4),
                })
    pairs.sort(key=lambda r: (-r["count"], r["true"], r["predicted"]))

    def rec_for(i):
        return {"test_index": int(i), "label": LABELS[int(y_test[i])],
                "predicted": LABELS[int(preds[i])],
                "predicted_prob": round(float(conf_pred[i]), 4),
                "true_class_prob": round(float(conf_true[i]), 4),
                "nonzero_voxels": int(occupancy[i])}

    err_idx = [int(i) for i in np.where(~correct)[0]]
    analysis = {
        "top_confusion_pairs": pairs,
        "errors": {"n": len(err_idx), "items": [rec_for(i) for i in err_idx]},
        "mean_predicted_prob_correct": round(float(conf_pred[correct].mean()), 4) if correct.any() else None,
        "mean_predicted_prob_incorrect": round(float(conf_pred[~correct].mean()), 4) if (~correct).any() else None,
        "nonzero_voxel_mean_correct": round(float(occupancy[correct].mean()), 1) if correct.any() else None,
        "nonzero_voxel_mean_incorrect": round(float(occupancy[~correct].mean()), 1) if (~correct).any() else None,
        "left_right_confusions": int(sum(
            cm[a, b] for a, b in [(1, 2), (2, 1), (3, 4), (4, 3), (7, 8), (8, 7)])),
    }

    record = {"split": "test", **metrics, "failure_analysis": analysis}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")

    # --- packaged samples, predictions re-measured from the .npy files ---
    os.makedirs(args.samples_dir, exist_ok=True)
    picks = {}
    if err_idx:
        picks["first_error"] = err_idx[0]
        picks["most_confident_error"] = max(err_idx, key=lambda i: conf_pred[i])
    corr_idx = [int(i) for i in np.where(correct)[0]]
    if corr_idx:
        picks["most_confident_correct"] = max(corr_idx, key=lambda i: conf_pred[i])
        picks["median_correct"] = corr_idx[len(corr_idx) // 2]
    index = {"selection_rule": (
        "first misclassified volume, most confident misclassified volume, "
        "most confident correct volume, and median correct volume in test "
        "order — from the packaged-weights evaluation in "
        "assets/evaluation_results.json"),
        "samples": []}
    for role, i in picks.items():
        fname = f"{role}_{i}.npy"
        np.save(os.path.join(args.samples_dir, fname), x_test[i])
        r = predict(model, os.path.join(args.samples_dir, fname))
        index["samples"].append({
            "file": fname, "role": role, "test_index": int(i),
            "label": LABELS[int(y_test[i])],
            "predicted": r["label"],
            "probabilities": {k: round(v, 4) for k, v in r["probabilities"].items()},
            "nonzero_voxels": int(occupancy[i]),
        })
        print(f"sample {fname}: label={LABELS[int(y_test[i])]} predicted={r['label']}")
    with open(os.path.join(args.samples_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {os.path.join(args.samples_dir, 'index.json')}")


if __name__ == "__main__":
    raise SystemExit(main())
