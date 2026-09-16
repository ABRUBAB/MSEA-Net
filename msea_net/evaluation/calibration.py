from typing import Dict, List, Tuple
import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width confidence bins.
    ECE = sum_{b=1}^B ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def split_conformal_prediction(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Split Conformal Prediction (SCP) for distribution-free finite-sample coverage guarantee.
    
    Non-conformity score: s_i = 1 - f(x_i)_{y_i}
    Target coverage: 1 - alpha (e.g., 95% coverage for alpha=0.05)
    """
    n_cal = len(cal_labels)
    # Compute non-conformity scores on calibration set
    cal_scores = 1.0 - cal_probs[np.arange(n_cal), cal_labels]

    # Conformal quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    q_level = min(1.0, q_level)
    q_hat = np.quantile(cal_scores, q_level, method='higher')

    # Construct prediction sets on test set: C(x) = {y : 1 - p_y <= q_hat}
    prediction_sets = test_probs >= (1.0 - q_hat)
    set_sizes = np.sum(prediction_sets, axis=1)

    # Empirical coverage on test set
    n_test = len(test_labels)
    contains_true = prediction_sets[np.arange(n_test), test_labels]
    empirical_coverage = float(np.mean(contains_true))
    mean_set_size = float(np.mean(set_sizes))

    return {
        'target_coverage': 1.0 - alpha,
        'empirical_coverage': empirical_coverage,
        'mean_set_size': mean_set_size,
        'quantile_threshold': float(q_hat),
        'single_class_pct': float(np.mean(set_sizes == 1)),
        'empty_set_pct': float(np.mean(set_sizes == 0)),
    }


def selective_prediction_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    threshold_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Evaluate Selective Classification (Risk-Coverage trade-off).
    Sorts samples by evidential uncertainty and computes accuracy as high-uncertainty
    predictions are progressively abstained from clinical decision-making.
    """
    preds = np.argmax(probs, axis=1)
    n_total = len(labels)

    # Sort indices by uncertainty ascending (lowest uncertainty first)
    sort_idx = np.argsort(uncertainties)
    sorted_preds = preds[sort_idx]
    sorted_labels = labels[sort_idx]

    coverages = np.linspace(0.1, 1.0, threshold_steps)
    accuracies = []

    for cov in coverages:
        cutoff = max(1, int(cov * n_total))
        acc = np.mean(sorted_preds[:cutoff] == sorted_labels[:cutoff])
        accuracies.append(acc)

    return {
        'coverages': coverages,
        'abstention_rates': 1.0 - coverages,
        'accuracies': np.array(accuracies),
    }
