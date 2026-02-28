# src/utils/io.py
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

def make_run_dir(base_dir: str | Path = "results", run_name: str | None = None) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def dump_config(run_dir: Path, config: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p