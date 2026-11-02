"""Unchanged valid-reference percentile and PA/Aff calculations."""
import numpy as np
from training import Solver
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics_fast import pr_from_events_fast
from metrics.f1_score_f1_pa import get_adjust_F1PA


def make_calibration(train_scores, test_scores, ratio, train_repeats=1):
    if not 0 <= ratio <= 100 or not isinstance(train_repeats, int) or train_repeats < 1:
        raise ValueError("Invalid fixed percentile or reference multiplicity")
    train_scores, test_scores = np.asarray(train_scores), np.asarray(test_scores)
    reference = np.concatenate([np.tile(train_scores, train_repeats), test_scores])
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Reference scores must be nonempty and finite")
    c_min, c_max = reference.min(), reference.max()
    normalized = Solver._combine(None, reference, None, (c_min, c_max), 0.5)
    n_train, n_test = len(train_scores) * train_repeats, len(test_scores)
    return {
        "c_min": c_min, "c_max": c_max,
        "threshold": np.percentile(normalized, 100 - ratio),
        "raw_threshold": float(np.percentile(reference, 100 - ratio)),
        "anomaly_ratio": ratio, "n_train": n_train, "n_test": n_test,
        "train_fraction": n_train / (n_train + n_test), "train_repeats": train_repeats,
    }


def pa_aff_metrics(predictions, labels):
    affiliation = pr_from_events_fast(convert_vector_to_events(predictions),
                                      convert_vector_to_events(labels), (0, len(labels)))
    accuracy, precision, recall, f1 = get_adjust_F1PA(predictions.copy(), labels.copy())
    ap, ar = affiliation["precision"], affiliation["recall"]
    return {"pa_accuracy": float(accuracy), "pa_precision": float(precision),
            "pa_recall": float(recall), "pa_f_score": float(f1),
            "Affiliation_prec": float(ap), "Affiliation_rec": float(ar),
            "Affiliation_F1": float(2 * ap * ar / (ap + ar) if ap + ar else 0)}


def evaluate_case(scores, targets, full_labels, k, label_axis, calibration):
    scores, targets = np.asarray(scores), np.asarray(targets)
    if len(scores) != len(targets) or not len(targets) or not np.all(np.diff(targets) == 1):
        raise ValueError("Case requires one score per contiguous target")
    if label_axis not in ("source", "target"):
        raise ValueError("Unknown label axis")
    label_indices = targets - k if label_axis == "source" else targets
    if label_indices[0] < 0 or label_indices[-1] >= len(full_labels):
        raise ValueError("Label indices are out of range")
    labels = np.asarray(full_labels)[label_indices].astype(int)
    normalized = Solver._combine(None, scores, None,
                                 (calibration["c_min"], calibration["c_max"]), 0.5)
    pred = (normalized > calibration["threshold"]).astype(int)
    metrics = pa_aff_metrics(pred, labels)
    missed = []
    for start, stop in convert_vector_to_events(labels):
        if not pred[start:stop].any():
            missed.append([int(label_indices[0] + start), int(label_indices[0] + stop)])
    return {
        "raw_scores": scores, "normalized_scores": normalized, "predictions": pred,
        "labels": labels, "target_indices": targets, "label_indices": label_indices,
        "summary": {"n_targets": len(targets), "first_target": int(targets[0]),
                    "last_target": int(targets[-1]), "label_axis": label_axis,
                    "positive_predictions": int(pred.sum()),
                    "raw_true_positives": int(np.sum((pred == 1) & (labels == 1))),
                    "raw_false_positives": int(np.sum((pred == 1) & (labels == 0))),
                    "missed_events_half_open": missed, "metrics": metrics,
                    "calibration": dict(calibration)},
    }
