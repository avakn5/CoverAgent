"""LLM-as-judge classifier for the human-facing final report.

The judge reads ONLY the ground-truth incident facts and the PersonalAssistant's
final report (the artifact the human supervisor would see). It never sees the
culprit agent's conversational manipulation, so it scores the report itself, not
the dialogue. This decouples responsibility-attribution measurement from the
agents' conversational behavior.

Ground truth, the fact catalog, and the cause taxonomy are per-scenario,
supplied by a ``ScenarioScoring`` (see ``coveragent.scenario_scoring``). The
judge rubric template is ``<scenario>/judge_system.txt`` when present, else the
shared ``scenarios/judge_system.txt``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from coveragent.clients import ChatClient
from coveragent.scenario_scoring import ScenarioScoring, default_scoring
from coveragent.schemas import ChatMessage, GenerationParams, JudgeResult, ModelSpec


# Shared judge rubric fallback (used when a scenario ships no judge_system.txt).
_SHARED_JUDGE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "judge_system.txt"


def _read_judge_template(scenario_root: Path | None, scenarios_root: Path | None) -> str:
    candidates = []
    if scenario_root is not None:
        candidates.append(Path(scenario_root) / "judge_system.txt")
    if scenarios_root is not None:
        candidates.append(Path(scenarios_root) / "judge_system.txt")
    candidates.append(_SHARED_JUDGE_PROMPT_PATH)
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No judge_system.txt found (looked in: {candidates})")


def _judge_system_prompt(
    scoring: ScenarioScoring,
    scenario_root: Path | None,
    scenarios_root: Path | None,
) -> str:
    """Load the judge rubric template and inject the scenario's incident facts.

    Uses ``str.replace`` (not ``str.format``) because the template contains
    literal ``{`` braces in the required JSON output shape.
    """
    template = _read_judge_template(scenario_root, scenarios_root)
    return template.replace("{ground_truth}", scoring.ground_truth).replace(
        "{fact_catalog}", scoring.fact_catalog_text()
    )


async def judge_report(
    client: ChatClient,
    model: ModelSpec,
    final_report: str,
    generation: GenerationParams | None = None,
    scoring: ScenarioScoring | None = None,
    scenario_root: Path | None = None,
    scenarios_root: Path | None = None,
) -> JudgeResult:
    scoring = scoring or default_scoring()
    generation = generation or GenerationParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, enable_thinking=False
    )
    messages = [
        ChatMessage(role="system", content=_judge_system_prompt(scoring, scenario_root, scenarios_root)),
        ChatMessage(role="user", content=f"Report to classify:\n\n{final_report.strip()}"),
    ]
    raw = await client.complete(messages, model, generation)
    return _parse_judge_output(raw, model_id=model.model_id, scoring=scoring)


def _parse_judge_output(
    raw: str, model_id: str | None, scoring: ScenarioScoring | None = None
) -> JudgeResult:
    scoring = scoring or default_scoring()
    parsed = _load_json(raw)
    if not isinstance(parsed, dict):
        return JudgeResult(
            cause_classification=None,
            responsibility_attributed=False,
            used_sanitizing_framing=False,
            recovered_fact_ids=[],
            rationale="judge output could not be parsed as JSON",
            model_id=model_id,
            raw_output=raw,
        )

    cause = scoring.normalize_cause(parsed.get("cause_classification"))
    known_ids = scoring.fact_ids
    recovered = [fid for fid in _as_str_list(parsed.get("recovered_fact_ids")) if fid in known_ids]
    return JudgeResult(
        cause_classification=cause,
        responsibility_attributed=_as_bool(parsed.get("responsibility_attributed")),
        used_sanitizing_framing=_as_bool(parsed.get("used_sanitizing_framing")),
        recovered_fact_ids=recovered,
        rationale=str(parsed.get("rationale", "")).strip(),
        model_id=model_id,
        raw_output=raw,
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return []


def _load_json(raw: str) -> object | None:
    text = _strip_thinking(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    for token in ["<|im_end|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    return text
