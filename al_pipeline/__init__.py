"""Active-learning utilities for ConfAL-WM.

This package is intentionally separate from the existing EVAC/C3 evaluation
scripts. The active-learning protocol must not use future GT or oracle errors
when scoring and selecting candidate pool samples.
"""

