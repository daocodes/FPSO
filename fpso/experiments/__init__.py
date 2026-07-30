"""Command-line entry points: cache building, running the matrix, analysis."""

from fpso.experiments.runner import load_panel, run_arm, run_matrix

__all__ = ["load_panel", "run_arm", "run_matrix"]
