from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(confidence, correct, bins=10) -> float:
    confidence, correct = np.asarray(confidence), np.asarray(correct).astype(float)
    if len(confidence) == 0: return 0.0
    edges, ece = np.linspace(0.0, 1.0, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if sel.any(): ece += sel.mean() * abs(confidence[sel].mean() - correct[sel].mean())
    return float(ece)


def normalized_entropy(probabilities) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def review_decision(quality: dict, confidence: float, coverage: float, entropy: float, thresholds=None) -> dict:
    thresholds = thresholds or {"confidence": 0.55, "coverage": 0.20, "entropy": 0.82}
    reasons = list(quality.get("flags", []))
    if confidence < thresholds["confidence"]: reasons.append("low_confidence")
    if coverage < thresholds["coverage"]: reasons.append("low_predicted_coverage")
    if entropy > thresholds["entropy"]: reasons.append("high_entropy")
    return {"status": "needs_review" if reasons else "accepted", "reasons": sorted(set(reasons)), "thresholds": thresholds}
