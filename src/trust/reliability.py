from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute temperature-scaled softmax probabilities."""
    z = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(confidence, correct, bins: int = 10) -> dict:
    """Compute Expected Calibration Error (ECE) and binned accuracy metrics."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return {
            "ece": 0.0,
            "mean_confidence": 0.0,
            "accuracy": 0.0,
            "sample_count": 0,
            "bins": [],
        }

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_details = []
    total_samples = len(confidence)

    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence >= lo) & (confidence <= hi if hi == 1.0 else confidence < hi)
        count = int(sel.sum())
        if count > 0:
            bin_conf = float(confidence[sel].mean())
            bin_acc = float(correct[sel].mean())
            weight = count / total_samples
            ece += weight * abs(bin_conf - bin_acc)
            bin_details.append({
                "range": [round(float(lo), 2), round(float(hi), 2)],
                "count": count,
                "mean_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
            })
        else:
            bin_details.append({
                "range": [round(float(lo), 2), round(float(hi), 2)],
                "count": 0,
                "mean_confidence": 0.0,
                "accuracy": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "sample_count": total_samples,
        "bins": bin_details,
    }


def normalized_entropy(probabilities) -> float:
    """Normalized Shannon entropy in [0, 1] for a discrete distribution."""
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def prediction_uncertainty(scores) -> float:
    """Compute prediction uncertainty score in [0, 1] from detection confidence scores.

    Uses normalized binary entropy H(s) = -[s log2(s) + (1-s) log2(1-s)].
    Scores near 0.5 yield maximum uncertainty (~1.0), while scores near 1.0 yield low uncertainty.
    """
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return 1.0
    s = np.clip(s, 1e-6, 1.0 - 1e-6)
    h = -(s * np.log2(s) + (1.0 - s) * np.log2(1.0 - s))
    return float(np.clip(np.mean(h), 0.0, 1.0))


def review_decision(quality: dict, confidence: float, coverage: float, uncertainty: float,
                    thresholds: dict | None = None) -> dict:
    """Evaluate whether an analysis result meets trustworthiness criteria or requires review."""
    default_thresholds = {
        "confidence": 0.22,      # Minimum mean confidence across detections
        "coverage": 0.05,        # Minimum benthic coverage required
        "uncertainty": 0.90,     # Maximum allowed prediction uncertainty
    }
    thresholds = {**default_thresholds, **(thresholds or {})}

    quality_flags = list(quality.get("flags", []))
    reliability_flags = []

    if confidence < thresholds["confidence"]:
        reliability_flags.append("low_confidence")
    if coverage < thresholds["coverage"]:
        reliability_flags.append("low_predicted_coverage")
    if uncertainty > thresholds["uncertainty"]:
        reliability_flags.append("high_uncertainty")

    all_reasons = sorted(set(quality_flags + reliability_flags))
    status = "needs_review" if all_reasons else "accepted"

    return {
        "status": status,
        "reasons": all_reasons,
        "quality_flags": quality_flags,
        "reliability_flags": reliability_flags,
        "thresholds": thresholds,
    }
