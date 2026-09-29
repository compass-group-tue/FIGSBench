"""Score-only first pass for calibrated validation (HR v11). No rule tags."""

from __future__ import annotations

from rule_guided_judging import run as base
from rule_guided_judging.hr_v11_common import CV_SCORE_PROMPT_VERSION, configure_axis_runner

PROMPT_VERSION = CV_SCORE_PROMPT_VERSION


def configure_base_runner() -> None:
    configure_axis_runner("calibrated_validation", score_only=True)
    base.PROMPT_VERSION = PROMPT_VERSION


def main() -> int:
    configure_base_runner()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
