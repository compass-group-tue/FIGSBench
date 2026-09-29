from __future__ import annotations

import pytest

from rule_guided_judging import hr_v11_common as common
from rule_guided_judging import hr_v11_cv_rules, hr_v11_syc_rules
from rule_guided_judging import run as base


@pytest.mark.parametrize("axis", ["sycophancy", "calibrated_validation"])
def test_frozen_score_prompt_hash_is_enforced(axis: str) -> None:
    common.configure_axis_runner(axis, score_only=True)
    current = base.sha256_text(base.load_system_prompt())
    assert base.verify_judge_lock(
        model="local-glm-5.3-flash", hashes={"system_prompt": current}
    )["status"] == "frozen"
    with pytest.raises(ValueError, match="frozen score prompt hash mismatch"):
        base.verify_judge_lock(
            model="local-glm-5.3-flash", hashes={"system_prompt": "tampered"}
        )


@pytest.mark.parametrize(
    "module,label",
    [
        (hr_v11_syc_rules, "sycophancy"),
        (hr_v11_cv_rules, "CV"),
    ],
)
def test_frozen_rules_prompt_hash_is_enforced(module: object, label: str) -> None:
    module.configure_rules_runner()  # type: ignore[attr-defined]
    current = base.sha256_text(base.load_system_prompt())
    assert base.verify_judge_lock(
        model="local-glm-5.3-flash", hashes={"system_prompt": current}
    )["status"] == "frozen"
    with pytest.raises(ValueError, match=f"frozen {label} rules prompt hash mismatch"):
        base.verify_judge_lock(
            model="local-glm-5.3-flash", hashes={"system_prompt": "tampered"}
        )
