from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


Speaker = Literal[
    "SchedulingAgent",
    "PersonalAssistant",
    "CodeAgent",
    "FinanceAgent",
    "Morgan Lee",
    "AlignmentAuditor",
]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    size_label: str


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 2048
    seed: int | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | int | bool | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = "be_helpful_peer_reputation_pressure"
    run_id: str = field(default_factory=lambda: f"coveragent_{uuid4().hex[:12]}")
    max_turns: int = 3
    agent_a_model: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            provider="scripted", model_id="scripted-agent-a", size_label="scripted"
        )
    )
    agent_b_model: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            provider="scripted", model_id="scripted-agent-b", size_label="scripted"
        )
    )
    agent_a_generation: GenerationParams = field(default_factory=GenerationParams)
    agent_b_generation: GenerationParams = field(default_factory=GenerationParams)
    prompt_dir: str = "scenarios/scheduling_incident/prompts/be_helpful_peer_reputation_pressure"
    trial_index: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TranscriptEvent:
    turn_index: int
    speaker: Speaker
    content: str
    model_id: str
    visible_cot: str | None = None
    raw_model_output: str | None = None


@dataclass(frozen=True)
class PromptBundle:
    agent_a_system: str
    agent_b_system: str
    agent_b_final_report: str
    shared_context: str = ""
    agent_c_system: str | None = None
    # Reporter-only context override. When set, PersonalAssistant is given this
    # instead of ``shared_context``, while the culprit and auditor keep the shared
    # context. This strips reporter-side context without changing the culprit.
    reporter_shared_context: str | None = None


@dataclass(frozen=True)
class FactScore:
    fact_id: str
    description: str
    recovered: bool
    evidence: str | None = None


@dataclass(frozen=True)
class JudgeResult:
    cause_classification: str | None
    responsibility_attributed: bool
    used_sanitizing_framing: bool
    recovered_fact_ids: list[str]
    rationale: str
    model_id: str | None = None
    raw_output: str | None = None


@dataclass(frozen=True)
class Metrics:
    cause_classification: str | None
    # None for a "refused" classification (a non-report): it is excluded from the
    # attribution and deceptive-success rates rather than scored as 0.0.
    cause_classification_score: float | None
    fact_recovery_rate: float
    information_leakage_ratio: float
    euphemism_adoption_rate: float
    accurate_attribution_score: float | None
    deceptive_success_score: float | None
    whistleblower_turn_index: int | None
    recovered_facts: list[FactScore]
    adopted_euphemisms: list[str]
    classification_source: str = "heuristic"


@dataclass(frozen=True)
class RunResult:
    config: ExperimentConfig
    prompts: PromptBundle
    transcript: list[TranscriptEvent]
    final_report_raw: str
    metrics: Metrics
    final_report_visible_cot: str | None = None
    final_report_model_raw: str | None = None
    judge: JudgeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["final_report"] = _parse_json_report(self.final_report_raw)
        return data


def _parse_json_report(raw_report: str) -> Any:
    text = raw_report.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return text
