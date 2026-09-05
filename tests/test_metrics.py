from __future__ import annotations

from t2i_distill.metrics import fit_alpha, normalized_entropy, probability_distribution, sharpen_distribution, total_variation


def test_fit_alpha_recovers_sharpening_direction() -> None:
    teacher = {"head": 0.7, "tail": 0.3}
    student = sharpen_distribution(teacher, 2.0)
    fit = fit_alpha(teacher, student)
    assert 1.8 <= fit["alpha"] <= 2.2
    assert fit["tv_student_to_pred"] < 0.02


def test_distribution_metrics_are_bounded() -> None:
    dist = probability_distribution(["a", "a", "b"], ["a", "b"])
    assert dist == {"a": 2 / 3, "b": 1 / 3}
    assert 0.0 <= normalized_entropy(dist) <= 1.0
    assert total_variation({"a": 1.0}, {"a": 0.5, "b": 0.5}) == 0.5
