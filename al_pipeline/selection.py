#!/usr/bin/env python3
"""Compatibility entry point for AL selection utilities.

The implementation lives in task_prescreen.py for now; this wrapper gives the
pipeline a method-neutral name while keeping old task_prescreen.py commands
working.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from al_pipeline.task_prescreen import main


if __name__ == "__main__":
    main()
