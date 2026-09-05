from __future__ import annotations

import pytest

from t2i_distill.power import audit_statistical_power


def test_power_audit_computes_mde_and_bonferroni_gate() -> None:
    audit = audit_statistical_power(
        {
            "overall_pass": True,
            "model_scope": "primary",
            "rq_pair_design_rows": [
                {
                    "rq": "rq1_sharpening_fit",
                    "comparison_id": "teacher__student",
                    "family": "unit",
                    "planned_matched_cells_by_benchmark": {"grade": 400, "dimcim": 900},
                }
            ],
        },
        min_matched_cells=100,
    )

    assert audit["overall_pass"] is True
    assert audit["benchmark_test_count"] == 2
    assert audit["bonferroni_alpha"] == pytest.approx(0.025)
    assert audit["rq_power_summary"][0]["min_planned_matched_cells"] == 400
    assert audit["power_rows"][0]["two_proportion_mde_bonferroni"] > audit["power_rows"][0]["two_proportion_mde_nominal"]


def test_power_audit_flags_underpowered_cells() -> None:
    audit = audit_statistical_power(
        {
            "overall_pass": True,
            "rq_pair_design_rows": [
                {
                    "rq": "safety_inheritance",
                    "comparison_id": "teacher__student",
                    "family": "unit",
                    "planned_matched_cells_by_benchmark": {"tiny": 12},
                }
            ],
        },
        min_matched_cells=100,
    )

    assert audit["overall_pass"] is False
    assert audit["gates"]["minimum_matched_cells"]["pass"] is False
    assert audit["power_rows"][0]["design_resolution"] == "underpowered"

def test_power_audit_can_scope_to_configured_primary_rqs() -> None:
    audit = audit_statistical_power(
        {
            "overall_pass": True,
            "rq_pair_design_rows": [
                {
                    "rq": "safety_inheritance",
                    "comparison_id": "teacher__student",
                    "family": "unit",
                    "planned_matched_cells_by_benchmark": {"overt": 72},
                },
                {
                    "rq": "rq3_preference_capability",
                    "comparison_id": "teacher__student",
                    "family": "unit",
                    "planned_matched_cells_by_benchmark": {"t2isafety": 18},
                },
            ],
        },
        min_matched_cells=30,
        included_rqs={"safety_inheritance"},
    )

    assert audit["overall_pass"] is True
    assert audit["benchmark_test_count"] == 1
    assert audit["power_rows"][0]["rq"] == "safety_inheritance"
    assert audit["included_rqs"] == ["safety_inheritance"]

