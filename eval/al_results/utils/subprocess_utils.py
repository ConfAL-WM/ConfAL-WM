from __future__ import annotations

import subprocess
import sys
from typing import Sequence


def run_command(cmd: Sequence[str], cwd: str | None = None, env: dict | None = None, dry_run: bool = False) -> int:
    """Print and optionally execute a command.  Returns exit code."""
    print("[cmd]", " ".join(cmd), flush=True)
    if dry_run:
        print("[cmd] (dry-run, not executed)")
        return 0
    return subprocess.run(cmd, cwd=cwd, env=env).returncode
