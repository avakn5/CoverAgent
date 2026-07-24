from __future__ import annotations

import json
from pathlib import Path

from coveragent.schemas import RunResult


def write_run_result(result: RunResult, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{result.config.run_id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path

