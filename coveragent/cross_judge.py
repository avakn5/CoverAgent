"""Post-hoc cross-judge validation for saved run artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable

from coveragent.clients import ChatClient
from coveragent.judge import judge_report
from coveragent.metrics import compute_metrics
from coveragent.models import build_judge, safe_model_label
from coveragent.scenario_scoring import load_scoring_for_prompt_dir
from coveragent.schemas import GenerationParams, TranscriptEvent


def find_artifacts(
    *,
    output_root: str | Path,
    experiment: str | None = None,
    pair_dir: str | None = None,
    source_glob: str | None = None,
    limit: int | None = None,
    selection_mode: str = "stratified",
    selection_seed: int = 0,
) -> list[Path]:
    """Find saved run JSON artifacts, excluding validation outputs."""
    root = Path(output_root)
    if source_glob:
        paths = root.glob(source_glob)
    elif experiment:
        base = root / experiment
        if pair_dir:
            base = base / pair_dir
        paths = base.rglob("*.json")
    else:
        paths = root.rglob("*.json")

    matches = [
        path
        for path in sorted(paths)
        if path.is_file() and not _looks_like_validation_path(path)
    ]
    return select_artifacts(
        matches,
        limit=limit,
        selection_mode=selection_mode,
        selection_seed=selection_seed,
    )


def select_artifacts(
    paths: list[Path],
    *,
    limit: int | None,
    selection_mode: str = "stratified",
    selection_seed: int = 0,
) -> list[Path]:
    """Choose a deterministic validation subset from candidate artifacts."""
    if limit is None or limit <= 0 or len(paths) <= limit:
        return paths
    mode = selection_mode.strip().lower()
    if mode == "first":
        return paths[:limit]
    if mode == "random":
        return _seeded_sample(paths, limit=limit, seed=selection_seed)
    if mode != "stratified":
        raise ValueError("selection_mode must be one of: stratified, random, first")

    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(_artifact_original_cause(path), []).append(path)
    for label, group in groups.items():
        groups[label] = _seeded_sample(group, limit=len(group), seed=selection_seed + _stable_int(label))

    selected: list[Path] = []
    labels = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for label in labels:
            group = groups[label]
            if not group:
                continue
            selected.append(group.pop(0))
            if len(selected) >= limit:
                break
    return sorted(selected)


async def run_cross_judge_validation(
    *,
    client: ChatClient,
    output_root: str | Path,
    validation_root: str | Path,
    judge_size: str = "qwen-30b-a3b",
    judge_backend: str = "vllm",
    judge_enable_thinking: bool = True,
    judge_temperature: float | None = None,
    judge_top_p: float | None = None,
    judge_max_tokens: int = 4096,
    prompts_root: str | Path = "scenarios",
    experiment: str | None = None,
    pair_dir: str | None = None,
    source_glob: str | None = None,
    limit: int | None = None,
    selection_mode: str = "stratified",
    selection_seed: int = 0,
    concurrency: int = 1,
    skip_existing: bool = True,
    on_summary: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Re-score saved final reports with a separate judge model.

    The original artifacts are never modified. Each validation writes a compact
    JSON file under ``validation_root/<judge-label>/`` and returns a summary.
    """
    paths = find_artifacts(
        output_root=output_root,
        experiment=experiment,
        pair_dir=pair_dir,
        source_glob=source_glob,
        limit=limit,
        selection_mode=selection_mode,
        selection_seed=selection_seed,
    )
    if not paths:
        return []

    judge_client, judge_model, generation = build_judge(client, judge_size, judge_backend)
    if judge_client is None or judge_model is None or generation is None:
        raise ValueError("cross-judge validation requires an enabled judge_size")
    generation = _judge_generation(
        base=generation,
        enable_thinking=judge_enable_thinking,
        temperature=judge_temperature,
        top_p=judge_top_p,
        max_tokens=judge_max_tokens,
    )

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(path: Path) -> dict:
        async with sem:
            summary = await score_artifact(
                path=path,
                client=judge_client,
                judge_model=judge_model,
                generation=generation,
                validation_root=validation_root,
                prompts_root=prompts_root,
                skip_existing=skip_existing,
            )
            if on_summary is not None:
                on_summary(summary)
            return summary

    return list(await asyncio.gather(*[_bounded(path) for path in paths]))


async def score_artifact(
    *,
    path: str | Path,
    client: ChatClient,
    judge_model,
    generation: GenerationParams,
    validation_root: str | Path,
    prompts_root: str | Path,
    skip_existing: bool = True,
) -> dict:
    source_path = Path(path)
    validation_path = _validation_path(source_path, validation_root, judge_model.size_label, generation)
    if skip_existing and validation_path.exists():
        return _summary_from_validation(validation_path, skipped_existing=True)

    data = json.loads(source_path.read_text(encoding="utf-8"))
    final_report = str(data.get("final_report_raw") or "")
    if not final_report.strip():
        summary = {
            "source_path": str(source_path),
            "validation_path": str(validation_path),
            "error": "artifact has no final_report_raw",
        }
        _write_validation(validation_path, summary)
        return summary

    config = data.get("config") or {}
    metrics = data.get("metrics") or {}
    prompt_dir = _resolve_prompt_dir(config.get("prompt_dir"), prompts_root)
    scoring, scenario_root, scenarios_root = load_scoring_for_prompt_dir(prompt_dir)
    judge_result = await judge_report(
        client,
        judge_model,
        final_report,
        generation,
        scoring=scoring,
        scenario_root=scenario_root,
        scenarios_root=scenarios_root,
    )
    transcript = _transcript_from_data(data.get("transcript") or [])
    cross_metrics = compute_metrics(final_report, transcript, judge_result=judge_result, scoring=scoring)
    original_cause = metrics.get("cause_classification")
    cross_cause = cross_metrics.cause_classification

    validation = {
        "validation_id": validation_path.stem,
        "source_path": str(source_path),
        "source_run_id": config.get("run_id", source_path.stem),
        "source_experiment": config.get("experiment_id"),
        "source_agent_a_size": (config.get("agent_a_model") or {}).get("size_label"),
        "source_agent_b_size": (config.get("agent_b_model") or {}).get("size_label"),
        "source_trial_index": config.get("trial_index"),
        "original": {
            "judge_model_id": (data.get("judge") or {}).get("model_id"),
            "cause_classification": original_cause,
            "classification_source": metrics.get("classification_source"),
            "accurate_attribution_score": metrics.get("accurate_attribution_score"),
            "fact_recovery_rate": metrics.get("fact_recovery_rate"),
        },
        "cross_judge": {
            "judge_size": judge_model.size_label,
            "judge_model_id": judge_model.model_id,
            "enable_thinking": generation.enable_thinking,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "result": asdict(judge_result),
        },
        "cross_metrics": asdict(cross_metrics),
        "agreement": {
            "cause_classification": original_cause == cross_cause,
            "accurate_attribution_score": metrics.get("accurate_attribution_score")
            == cross_metrics.accurate_attribution_score,
        },
    }
    _write_validation(validation_path, validation)
    return _summary_from_validation(validation_path)


def summarize_validations(summaries: list[dict]) -> dict:
    scored = [summary for summary in summaries if "error" not in summary]
    changed = [summary for summary in scored if not summary.get("agrees_on_cause")]
    return {
        "validated": len(scored),
        "errors": len(summaries) - len(scored),
        "cause_agreement_rate": round((len(scored) - len(changed)) / len(scored), 4) if scored else None,
        "changed": len(changed),
        "original_counts": _counts(summary.get("original_cause") for summary in scored),
        "cross_judge_counts": _counts(summary.get("cross_cause") for summary in scored),
    }


def _judge_generation(
    *,
    base: GenerationParams,
    enable_thinking: bool,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int,
) -> GenerationParams:
    if temperature is None:
        temperature = 0.6 if enable_thinking else base.temperature
    if top_p is None:
        top_p = 0.95 if enable_thinking else base.top_p
    return GenerationParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=0,
        enable_thinking=enable_thinking,
        reasoning_effort=base.reasoning_effort,
    )


def _validation_path(
    source_path: Path,
    validation_root: str | Path,
    judge_size: str,
    generation: GenerationParams,
) -> Path:
    thinking = "thinking" if generation.enable_thinking else "no-thinking"
    judge_label = f"{safe_model_label(judge_size)}-{thinking}"
    return Path(validation_root) / judge_label / f"{source_path.stem}.json"


def _summary_from_validation(path: Path, *, skipped_existing: bool = False) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "error" in data:
        summary = dict(data)
    else:
        summary = {
            "source_run_id": data.get("source_run_id"),
            "source_path": data.get("source_path"),
            "validation_path": str(path),
            "original_cause": (data.get("original") or {}).get("cause_classification"),
            "cross_cause": (data.get("cross_metrics") or {}).get("cause_classification"),
            "original_score": (data.get("original") or {}).get("accurate_attribution_score"),
            "cross_score": (data.get("cross_metrics") or {}).get("accurate_attribution_score"),
            "agrees_on_cause": (data.get("agreement") or {}).get("cause_classification"),
            "judge_model_id": ((data.get("cross_judge") or {}).get("judge_model_id")),
        }
    if skipped_existing:
        summary["skipped_existing"] = True
    return summary


def _write_validation(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve_prompt_dir(prompt_dir: object, prompts_root: str | Path) -> Path:
    if isinstance(prompt_dir, str) and prompt_dir.strip():
        candidate = Path(prompt_dir)
        if candidate.exists():
            return candidate
        marker = "scenarios/"
        if marker in prompt_dir:
            relative = prompt_dir.split(marker, 1)[1]
            mapped = Path(prompts_root) / relative
            if mapped.exists():
                return mapped
            return mapped
        mapped = Path(prompts_root) / prompt_dir
        if mapped.exists():
            return mapped
    raise FileNotFoundError(f"Could not resolve prompt_dir={prompt_dir!r} under {prompts_root!r}")


def _transcript_from_data(raw_events: list[object]) -> list[TranscriptEvent]:
    events = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        allowed = {
            "turn_index",
            "speaker",
            "content",
            "model_id",
            "visible_cot",
            "raw_model_output",
        }
        kwargs = {key: raw.get(key) for key in allowed if key in raw}
        try:
            events.append(TranscriptEvent(**kwargs))
        except TypeError:
            continue
    return events


def _looks_like_validation_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool({"validation", "validations", "cross_judge"} & parts)


def _artifact_original_cause(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "__unreadable__"
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return "__missing_metrics__"
    cause = metrics.get("cause_classification")
    return str(cause) if cause is not None else "__none__"


def _seeded_sample(paths: list[Path], *, limit: int, seed: int) -> list[Path]:
    import random

    items = list(paths)
    random.Random(seed).shuffle(items)
    return items[:limit]


def _stable_int(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
