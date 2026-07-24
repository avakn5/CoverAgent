"""Model registry and naming helpers shared by every backend.

"""

from __future__ import annotations

from coveragent.schemas import GenerationParams, ModelSpec


MODEL_ALIASES = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
    "14B": "Qwen/Qwen3-14B",
    "32B": "Qwen/Qwen3-32B",
    "30B-A3B": "Qwen/Qwen3-30B-A3B",
    "235B-A22B": "Qwen/Qwen3-235B-A22B",
    "qwen-0.6b": "Qwen/Qwen3-0.6B",
    "qwen-1.7b": "Qwen/Qwen3-1.7B",
    "qwen-4b": "Qwen/Qwen3-4B",
    "qwen-8b": "Qwen/Qwen3-8B",
    "qwen-14b": "Qwen/Qwen3-14B",
    "qwen-32b": "Qwen/Qwen3-32B",
    "qwen-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "llama-7b": "NousResearch/Llama-2-7b-chat-hf",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gemma-9b": "google/gemma-2-9b-it",
    "gemma-27b": "google/gemma-2-27b-it",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "olmo-7b": "allenai/OLMo-2-1124-7B-Instruct",
}

# Fireworks serverless model paths (OpenAI-compatible ids). VERIFY current ids /
# availability on https://fireworks.ai/models — these change. Use `fw:<full/path>`
# to pass an id directly. Reasoning is captured for thinking-capable models.
FIREWORKS_MODELS = {
    "gpt-oss-20b": "accounts/fireworks/models/gpt-oss-20b",
    "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
    "deepseek-v4-flash": "accounts/fireworks/models/deepseek-v4-flash",
    "glm-5.2": "accounts/fireworks/models/glm-5p2",
    "glm5.2": "accounts/fireworks/models/glm-5p2",
    "qwen3.7-plus": "accounts/fireworks/models/qwen3p7-plus",
    "qwen3p7-plus": "accounts/fireworks/models/qwen3p7-plus",
    "235B-A22B": "accounts/fireworks/models/qwen3-235b-a22b",
    "qwen3-235b": "accounts/fireworks/models/qwen3-235b-a22b",
    "30B-A3B": "accounts/fireworks/models/qwen3-30b-a3b",
    "qwen3-30b-a3b": "accounts/fireworks/models/qwen3-30b-a3b",
    "8B": "accounts/fireworks/models/qwen3-8b",
    "qwen3-8b": "accounts/fireworks/models/qwen3-8b",
    "deepseek-r1": "accounts/fireworks/models/deepseek-r1",
    "deepseek-v3": "accounts/fireworks/models/deepseek-v3",
    "llama-70b": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "llama-8b": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "qwen2p5-72b": "accounts/fireworks/models/qwen2p5-72b-instruct",
}

OPENROUTER_MODELS = {
    "0.6B": "qwen/qwen3-0.6b",
    "qwen3-0.6b": "qwen/qwen3-0.6b",
    "1.7B": "qwen/qwen3-1.7b",
    "qwen3-1.7b": "qwen/qwen3-1.7b",
    "4B": "qwen/qwen3-4b",
    "qwen3-4b": "qwen/qwen3-4b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "glm-5.2": "z-ai/glm-5.2",
    "glm5.2": "z-ai/glm-5.2",
    "qwen3.7-plus": "qwen/qwen3.7-plus",
    "qwen3p7-plus": "qwen/qwen3.7-plus",
    "8B": "qwen/qwen3-8b",
    "qwen3-8b": "qwen/qwen3-8b",
    "14B": "qwen/qwen3-14b",
    "qwen3-14b": "qwen/qwen3-14b",
    "32B": "qwen/qwen3-32b",
    "qwen3-32b": "qwen/qwen3-32b",
    "235B-A22B": "qwen/qwen3-235b-a22b",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "mistral-medium-3.5": "mistralai/mistral-medium-3-5",
    "mistral-medium-3-5": "mistralai/mistral-medium-3-5",
}

SAIL_MODELS = {
    "glm-5.2": "zai-org/GLM-5.2-FP8",
    "glm5.2": "zai-org/GLM-5.2-FP8",
    "glm-5.1": "zai-org/GLM-5.1-FP8",
    "glm5.1": "zai-org/GLM-5.1-FP8",
}

JUDGE_DISABLED = {"", "none", "off"}


def known_model_names(backend: str = "transformers") -> list[str]:
    if backend == "fireworks":
        return sorted(FIREWORKS_MODELS) + ["fw:<accounts/.../model>"]
    if backend == "openrouter":
        return sorted(OPENROUTER_MODELS) + ["or:<provider/model>"]
    if backend == "sail":
        return sorted(SAIL_MODELS) + ["sail:<provider/model>"]
    return sorted(MODEL_ALIASES) + ["hf:<org/model>"]


def resolve_model_id(model_name: str, backend: str = "transformers") -> str:
    if backend == "fireworks":
        if model_name.startswith("fw:"):
            model_id = model_name[3:].strip()
            if model_id:
                return model_id
        if model_name in FIREWORKS_MODELS:
            return FIREWORKS_MODELS[model_name]
        raise ValueError(f"Unknown Fireworks model {model_name!r}. Choose from {known_model_names('fireworks')}")
    if backend == "openrouter":
        if model_name.startswith("or:"):
            model_id = model_name[3:].strip()
            if model_id:
                return model_id
        if model_name in OPENROUTER_MODELS:
            return OPENROUTER_MODELS[model_name]
        raise ValueError(f"Unknown OpenRouter model {model_name!r}. Choose from {known_model_names('openrouter')}")
    if backend == "sail":
        if model_name.startswith("sail:"):
            model_id = model_name[5:].strip()
            if model_id:
                return model_id
        if model_name in SAIL_MODELS:
            return SAIL_MODELS[model_name]
        raise ValueError(f"Unknown Sail model {model_name!r}. Choose from {known_model_names('sail')}")
    if model_name.startswith("hf:"):
        model_id = model_name[3:].strip()
        if model_id:
            return model_id
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    raise ValueError(f"Unknown model {model_name!r}. Choose from {known_model_names()}")


def provider_name(model_id: str) -> str:
    if model_id.startswith("accounts/"):
        return "fireworks"
    if model_id.startswith(("openai/", "qwen/", "z-ai/", "anthropic/", "google/")):
        return "openrouter"
    if model_id.startswith("zai-org/"):
        return "sail"
    if model_id.startswith("Qwen/"):
        return "qwen"
    return "hf"


def supports_qwen_thinking(model_id: str) -> bool:
    return model_id.startswith("Qwen/Qwen3")


def requires_trust_remote_code(model_id: str) -> bool:
    return model_id.startswith("Qwen/") or model_id.startswith("allenai/")


def safe_model_label(model_name: str) -> str:
    label = model_name[3:] if model_name.startswith("hf:") else model_name
    for char in ["/", ":", " "]:
        label = label.replace(char, "-")
    return label


def pair_dir_name(agent_a_size: str, agent_b_size: str) -> str:
    return f"{safe_model_label(agent_a_size)}-{safe_model_label(agent_b_size)}"


def prompt_dir_name(experiment: str) -> str:
    scheduling_root = "scheduling_incident/prompts"
    if experiment in {
        "be_helpful_peer_reputation_pressure",
        "peer_reputation_pressure",
    } or experiment.startswith("exp1"):
        return f"{scheduling_root}/be_helpful_peer_reputation_pressure"
    if experiment in {
        "no_guardrails_peer_reputation_pressure",
        "minimal_guardrails_peer_rep_pressure",
    }:
        return f"{scheduling_root}/no_guardrails_peer_reputation_pressure"
    if experiment == "no_pressure_control":
        return f"{scheduling_root}/no_pressure_control"

    coding_root = "coding_incident/prompts"
    if experiment == "coding_no_pressure_control":
        return f"{coding_root}/no_pressure_control"
    if experiment in {"coding_be_helpful_peer_reputation_pressure", "coding_peer_reputation_pressure"}:
        return f"{coding_root}/be_helpful_peer_reputation_pressure"
    if experiment in {
        "coding_no_guardrails_peer_reputation_pressure",
        "coding_minimal_guardrails_peer_rep_pressure",
    }:
        return f"{coding_root}/no_guardrails_peer_reputation_pressure"

    financial_root = "financial_incident/prompts"
    if experiment == "financial_no_pressure_control":
        return f"{financial_root}/no_pressure_control"
    if experiment in {"financial_be_helpful_peer_reputation_pressure", "financial_peer_reputation_pressure"}:
        return f"{financial_root}/be_helpful_peer_reputation_pressure"
    if experiment in {
        "financial_no_guardrails_peer_reputation_pressure",
        "financial_minimal_guardrails_peer_rep_pressure",
    }:
        return f"{financial_root}/no_guardrails_peer_reputation_pressure"
    return experiment


def build_judge(client, judge_size: str, backend: str = "transformers"):
    """Build the (judge_client, judge_model, judge_generation) triple.

    Returns ``(None, None, None)`` when ``judge_size`` is empty/"none"/"off",
    which makes the run fall back to the deterministic heuristic scorer. The
    judge reuses the dialogue ``client`` (model caching means a judge equal to
    the agent size costs no extra memory); a larger judge loads an extra model
    on the same device. Greedy decoding keeps the judge deterministic.
    """
    if not judge_size or judge_size.lower() in JUDGE_DISABLED:
        return None, None, None
    judge_model_id = resolve_model_id(judge_size, backend)
    judge_model = ModelSpec(
        provider=provider_name(judge_model_id),
        model_id=judge_model_id,
        size_label=judge_size,
    )
    judge_generation = GenerationParams(temperature=0.0, top_p=1.0, max_tokens=1024, enable_thinking=False)
    return client, judge_model, judge_generation
