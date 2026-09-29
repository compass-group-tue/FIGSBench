"""Compatibility entry point for validating a public release checkout."""

from __future__ import annotations

from benchmark.data_schema import main


if __name__ == "__main__":
    raise SystemExit(main())
