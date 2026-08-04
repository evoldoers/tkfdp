"""Mood-light permutation-field model (paper 2b Sec. 4, simpler variant)."""
from .field_ctmc import build_field, transposition_distance, n_cycles, field_dims

__all__ = ["build_field", "transposition_distance", "n_cycles", "field_dims"]
