"""Finalized-rulebook, research-only SFT data generation."""

from .planning import DEFAULT_TARGET, build_assignments
from .rulebooks import load_finalized_rulebooks

__all__ = ["DEFAULT_TARGET", "build_assignments", "load_finalized_rulebooks"]
