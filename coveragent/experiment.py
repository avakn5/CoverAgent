"""Backend-agnostic experiment orchestration.

These helpers build an ExperimentConfig, run one dialogue (with the LLM judge
when enabled), write the JSON artifact, and return a compact summary dict. 
"""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from coveragent.clients import ChatClient
from coveragent.dialogue import run_dialogue
from coveragent.io import write_run_result
from coveragent.models import (
    build_judge,
    pair_dir_name,
    prompt_dir_name,
    provider_name,
    resolve_model_id,
    safe_model_label,
)
from coveragent.schemas import ExperimentConfig, GenerationParams, ModelSpec, RunResult


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(experiment: str, agent_a_size: str, agent_b_size: str, trial_index: int, timestamp: str) -> str:
    a = safe_model_label(agent_a_size)
    b = safe_model_label(agent_b_size)
    return f"{experiment}_A{a}_B{b}_t{trial_index}_{timestamp}"


def build_config(
    *,
    experiment: str,
    run_id: str,
    agent_a_size: str,
    agent_b_size: str,
    trial_index: int,
    max_turns: int,
    prompts_root: str,
    params: GenerationParams,
    backend: str = "transformers",
) -> ExperimentConfig:
    agent_a_model_id = resolve_model_id(agent_a_size, backend)
    agent_b_model_id = resolve_model_id(agent_b_size, backend)
    prompt_dir = f"{prompts_root.rstrip('/')}/{prompt_dir_name(experiment)}"
    return ExperimentConfig(
        experiment_id=experiment,
        run_id=run_id,
        max_turns=max_turns,
        prompt_dir=prompt_dir,
        trial_index=trial_index,
        agent_a_model=ModelSpec(provider=provider_name(agent_a_model_id), model_id=agent_a_model_id, size_label=agent_a_size),
        agent_b_model=ModelSpec(provider=provider_name(agent_b_model_id), model_id=agent_b_model_id, size_label=agent_b_size),
        agent_a_generation=params,
        agent_b_generation=params,
    )


def summarize(result: RunResult, output_path: Path, judge_size: str) -> dict:
    config = result.config
    metrics = result.metrics
    return {
        "run_id": config.run_id,
        "output_path": str(output_path),
        "experiment": config.experiment_id,
        "agent_a_size": config.agent_a_model.size_label,
        "agent_b_size": config.agent_b_model.size_label,
        "trial_index": config.trial_index,
        "judge_size": judge_size,
        "cause_classification": metrics.cause_classification,
        "classification_source": metrics.classification_source,
        "cause_classification_score": metrics.cause_classification_score,
        "accurate_attribution_score": metrics.accurate_attribution_score,
        "deceptive_success_score": metrics.deceptive_success_score,
        "fact_recovery_rate": metrics.fact_recovery_rate,
        "euphemism_adoption_rate": metrics.euphemism_adoption_rate,
    }


async def run_one(
    *,
    client: ChatClient,
    agent_a_size: str,
    agent_b_size: str,
    trial_index: int = 0,
    max_turns: int = 3,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    experiment: str = "be_helpful_peer_reputation_pressure",
    judge_size: str = "14B",
    judge_client: ChatClient | None = None,
    prompts_root: str = "scenarios",
    output_root: str | Path = "runs",
    timestamp: str | None = None,
    run_id: str | None = None,
    backend: str = "transformers",
    enable_thinking: bool | None = None,
    reasoning_effort: str | int | bool | None = None,
    seed_base: int = 0,
    judge_backend: str | None = None,
) -> tuple[RunResult, Path, dict]:
    """Run a single dialogue and write its artifact. ``judge_client`` defaults to
    reusing ``client`` (with ``judge_size`` selecting the judge model)."""
    timestamp = timestamp or utc_timestamp()
    run_id = run_id or make_run_id(experiment, agent_a_size, agent_b_size, trial_index, timestamp)
    params = GenerationParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        seed=seed_base + trial_index, enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    config = build_config(
        experiment=experiment,
        run_id=run_id,
        agent_a_size=agent_a_size,
        agent_b_size=agent_b_size,
        trial_index=trial_index,
        max_turns=max_turns,
        prompts_root=prompts_root,
        params=params,
        backend=backend,
    )

    effective_judge_backend = judge_backend or backend
    jc, judge_model, judge_generation = build_judge(judge_client or client, judge_size, effective_judge_backend)
    result = await run_dialogue(
        config,
        agent_a_client=client,
        agent_b_client=client,
        judge_client=jc,
        judge_model=judge_model,
        judge_generation=judge_generation,
    )

    output_dir = Path(output_root) / experiment / pair_dir_name(agent_a_size, agent_b_size)
    output_path = write_run_result(result, output_dir)
    return result, output_path, summarize(result, output_path, judge_size)


def existing_trial_summary(
    *,
    output_root: str | Path,
    experiment: str,
    agent_a_size: str,
    agent_b_size: str,
    trial_index: int,
    judge_size: str,
) -> dict | None:
    """Return a summary for an already-written trial, if present.

    Modal may preempt and restart a sequential batch from the same input. Run IDs
    include timestamps, so a restarted batch would otherwise duplicate completed
    trial indices. This helper makes resume jobs idempotent by trial index.
    """
    output_dir = Path(output_root) / experiment / pair_dir_name(agent_a_size, agent_b_size)
    pattern = (
        f"{experiment}_A{safe_model_label(agent_a_size)}_"
        f"B{safe_model_label(agent_b_size)}_t{trial_index}_*.json"
    )
    matches = sorted(output_dir.glob(pattern))
    if not matches:
        return None
    path = matches[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"run_id": path.stem, "output_path": str(path), "trial_index": trial_index, "skipped_existing": True}
    config = data.get("config", {})
    metrics = data.get("metrics", {})
    return {
        "run_id": config.get("run_id", path.stem),
        "output_path": str(path),
        "experiment": config.get("experiment_id", experiment),
        "agent_a_size": (config.get("agent_a_model") or {}).get("size_label", agent_a_size),
        "agent_b_size": (config.get("agent_b_model") or {}).get("size_label", agent_b_size),
        "trial_index": config.get("trial_index", trial_index),
        "judge_size": judge_size,
        "cause_classification": metrics.get("cause_classification"),
        "classification_source": metrics.get("classification_source"),
        "cause_classification_score": metrics.get("cause_classification_score"),
        "accurate_attribution_score": metrics.get("accurate_attribution_score"),
        "deceptive_success_score": metrics.get("deceptive_success_score"),
        "fact_recovery_rate": metrics.get("fact_recovery_rate"),
        "euphemism_adoption_rate": metrics.get("euphemism_adoption_rate"),
        "skipped_existing": True,
    }


async def run_same_size_trials(
    *,
    client: ChatClient,
    size: str,
    trials: int = 5,
    start_trial_index: int = 0,
    max_turns: int = 3,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    experiment: str = "be_helpful_peer_reputation_pressure",
    judge_size: str = "14B",
    judge_client: ChatClient | None = None,
    prompts_root: str = "scenarios",
    output_root: str | Path = "runs",
    on_summary=None,
    backend: str = "transformers",
    enable_thinking: bool | None = None,
    reasoning_effort: str | int | bool | None = None,
    seed_base: int = 0,
    concurrency: int = 1,
    skip_existing: bool = False,
    trial_delay_seconds: float = 0.0,
    judge_backend: str | None = None,
) -> list[dict]:
    """Run matched-size trials (SchedulingAgent size == PersonalAssistant size).

    ``concurrency`` > 1 runs trials with an ``asyncio`` semaphore. Use it for
    I/O-bound API backends and the vLLM backend; the shared-GPU transformers
    backend must stay at 1. A trial that errors is recorded as an error summary
    and does not abort the batch.
    """

    async def _trial(offset: int) -> dict:
        trial_index = start_trial_index + offset
        if skip_existing:
            existing = existing_trial_summary(
                output_root=output_root,
                experiment=experiment,
                agent_a_size=size,
                agent_b_size=size,
                trial_index=trial_index,
                judge_size=judge_size,
            )
            if existing is not None:
                if on_summary is not None:
                    on_summary(existing)
                return existing
        try:
            _, _, summary = await run_one(
                client=client,
                agent_a_size=size,
                agent_b_size=size,
                trial_index=trial_index,
                max_turns=max_turns,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                experiment=experiment,
                judge_size=judge_size,
                judge_client=judge_client,
                prompts_root=prompts_root,
                output_root=output_root,
                backend=backend,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                seed_base=seed_base,
                judge_backend=judge_backend,
            )
        except Exception as exc:  # keep the batch alive on a single bad trial
            summary = {"run_id": f"{experiment}_t{trial_index}", "error": repr(exc)}
            print(f"{summary['run_id']}: ERROR {summary['error']}")
            return summary
        if on_summary is not None:
            on_summary(summary)
        return summary

    if concurrency <= 1:
        summaries: list[dict] = []
        for offset in range(trials):
            summaries.append(await _trial(offset))
            if trial_delay_seconds > 0 and offset < trials - 1:
                await asyncio.sleep(trial_delay_seconds)
        return summaries

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(offset: int) -> dict:
        async with sem:
            return await _trial(offset)

    return list(await asyncio.gather(*[_bounded(offset) for offset in range(trials)]))
