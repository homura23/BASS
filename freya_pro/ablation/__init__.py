"""Controlled ablation contracts for the frozen CARLA R1 arm."""

from .r1_only import build_r1_only_plan, validate_r1_only_plan

__all__ = ["build_r1_only_plan", "validate_r1_only_plan"]
