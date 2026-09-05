from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Sequence


EPS = 1e-12


def probability_distribution(labels: Iterable[str], support: Sequence[str] | None = None) -> dict[str, float]:
    counts = Counter(label for label in labels if label != "")
    if support is None:
        support = sorted(counts)
    total = sum(counts[label] for label in support)
    if total == 0:
        return {label: 0.0 for label in support}
    return {label: counts[label] / total for label in support}


def normalized_entropy(dist: dict[str, float]) -> float:
    support = [p for p in dist.values() if p > 0]
    if len(dist) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in support)
    return entropy / math.log(len(dist))


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    support = set(p) | set(q)
    return 0.5 * sum(abs(p.get(label, 0.0) - q.get(label, 0.0)) for label in support)


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    support = set(p) | set(q)
    value = 0.0
    for label in support:
        pv = p.get(label, 0.0)
        if pv <= 0:
            continue
        value += pv * math.log(pv / max(q.get(label, 0.0), EPS))
    return value


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    support = set(p) | set(q)
    m = {label: 0.5 * (p.get(label, 0.0) + q.get(label, 0.0)) for label in support}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def hellinger(p: dict[str, float], q: dict[str, float]) -> float:
    support = set(p) | set(q)
    sq = sum((math.sqrt(p.get(label, 0.0)) - math.sqrt(q.get(label, 0.0))) ** 2 for label in support)
    return math.sqrt(sq) / math.sqrt(2.0)


def sharpen_distribution(teacher: dict[str, float], alpha: float) -> dict[str, float]:
    weights = {label: max(prob, 0.0) ** alpha for label, prob in teacher.items()}
    total = sum(weights.values())
    if total <= 0:
        return {label: 0.0 for label in teacher}
    return {label: value / total for label, value in weights.items()}


def fit_alpha(
    teacher: dict[str, float],
    student: dict[str, float],
    *,
    alpha_min: float = 0.25,
    alpha_max: float = 8.0,
    steps: int = 776,
) -> dict[str, float]:
    support = sorted(set(teacher) | set(student))
    t = _complete(teacher, support)
    s = _complete(student, support)
    best_alpha = alpha_min
    best_kl = float("inf")
    best_tv = float("inf")
    for idx in range(steps + 1):
        alpha = alpha_min + (alpha_max - alpha_min) * idx / steps
        pred = sharpen_distribution(t, alpha)
        kl = kl_divergence(s, pred)
        tv = total_variation(s, pred)
        if (kl, tv) < (best_kl, best_tv):
            best_alpha = alpha
            best_kl = kl
            best_tv = tv
    pred = sharpen_distribution(t, best_alpha)
    return {
        "alpha": best_alpha,
        "kl_student_to_pred": best_kl,
        "tv_student_to_pred": best_tv,
        "js_student_to_pred": js_divergence(s, pred),
        "hellinger_student_to_pred": hellinger(s, pred),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(values: Sequence[float], *, iterations: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int((1 - confidence) / 2 * iterations)
    hi_idx = int((1 + confidence) / 2 * iterations)
    return means[max(0, lo_idx)], means[min(iterations - 1, hi_idx)]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _ranks(x)
    ry = _ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / (den_x * den_y)


def _complete(dist: dict[str, float], support: Sequence[str]) -> dict[str, float]:
    total = sum(max(dist.get(label, 0.0), 0.0) for label in support)
    if total <= 0:
        return {label: 1.0 / len(support) for label in support} if support else {}
    return {label: max(dist.get(label, 0.0), 0.0) / total for label in support}


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + end - 1) / 2 + 1
        for original, _ in indexed[idx:end]:
            ranks[original] = rank
        idx = end
    return ranks
