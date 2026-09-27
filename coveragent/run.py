"""Local experiment runner.

Examples:
  python -m coveragent.run --size 4B --trials 5 --judge-size 14B
  python -m coveragent.run --agent-a-size 8B --agent-b-size 0.6B
"""

from __future__ import annotations

import argparse
import asyncio

from coveragent.experiment import run_one, run_same_size_trials


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CoverAgent experiments locally (no Modal required).")
    parser.add_argument("--experiment", default="be_helpful_peer_reputation_pressure")
    parser.add_argument("--prompts-root", default="scenarios")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="trials in flight at once (use for API/vLLM; keep 1 for transformers)")
    parser.add_argument("--trial-delay-seconds", type=float, default=0.0,
                        help="delay between sequential trials; useful for rate-limited API routes")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="trial t uses seed = seed_base + t (default 0)")
    parser.add_argument("--judge-size", default="14B", help='judge model alias, or "none" to use the heuristic')
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm", "fireworks", "openrouter", "sail", "modal"],
                        help="inference backend")
    parser.add_argument("--judge-backend", default=None, choices=["transformers", "vllm", "fireworks", "openrouter", "sail", "modal"],
                        help="optional backend for the judge; defaults to --backend")
    parser.add_argument("--no-thinking", action="store_true",
                        help="disable reasoning/CoT (default: on for thinking-capable models)")
    parser.add_argument("--reasoning-effort", default="medium",
                        help="Fireworks reasoning_effort for agents when thinking is enabled")
    # model selection
    parser.add_argument("--size", default=None, help="matched size for both agents; runs --trials trials")
    parser.add_argument("--agent-a-size", default=None, help="SchedulingAgent size (asymmetric single pair)")
    parser.add_argument("--agent-b-size", default=None, help="PersonalAssistant size (asymmetric single pair)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--start-trial-index", type=int, default=0)
    return parser


def _print_summary(summary: dict) -> None:
    print(
        f"{summary['run_id']}: "
        f"cause={summary['cause_classification']} "
        f"src={summary['classification_source']} "
        f"attribution={summary['accurate_attribution_score']} "
        f"deceptive={summary['deceptive_success_score']}"
    )


def _build_client(backend: str):
    if backend == "fireworks":
        from coveragent.backends import FireworksChatClient

        return FireworksChatClient()
    if backend == "openrouter":
        from coveragent.backends import OpenRouterChatClient

        return OpenRouterChatClient()
    if backend == "modal":
        from coveragent.backends import ModalChatClient

        return ModalChatClient()
    if backend == "sail":
        from coveragent.backends import SailChatClient

        return SailChatClient()
    if backend == "vllm":
        from coveragent.backends import VllmChatClient

        return VllmChatClient()
    from coveragent.backends import TransformersChatClient

    return TransformersChatClient()


async def _run(args: argparse.Namespace) -> None:
    client = _build_client(args.backend)
    judge_client = None
    if args.judge_backend is not None and args.judge_backend != args.backend and args.judge_size.lower() not in {"", "none", "off"}:
        judge_client = _build_client(args.judge_backend)
    enable_thinking = False if args.no_thinking else (True if args.backend in {"fireworks", "openrouter", "modal"} else None)
    reasoning_effort = None if args.no_thinking or args.backend not in {"fireworks", "openrouter", "modal"} else args.reasoning_effort
    concurrency = args.concurrency
    if args.backend == "transformers" and concurrency > 1:
        print("warning: --concurrency >1 is unsafe on the shared GPU; forcing 1 for transformers")
        concurrency = 1
    common = dict(
        max_turns=args.max_turns,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        experiment=args.experiment,
        judge_size=args.judge_size,
        prompts_root=args.prompts_root,
        output_root=args.output_root,
        backend=args.backend,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        seed_base=args.seed_base,
        judge_backend=args.judge_backend,
    )

    if args.agent_a_size and args.agent_b_size:
        _, path, summary = await run_one(
            client=client,
            agent_a_size=args.agent_a_size,
            agent_b_size=args.agent_b_size,
            trial_index=args.start_trial_index,
            judge_client=judge_client,
            **common,
        )
        print(f"Wrote {path}")
        _print_summary(summary)
        return

    size = args.size
    if not size:
        raise SystemExit("need --size, or both --agent-a-size and --agent-b-size")
    await run_same_size_trials(
        client=client,
        size=size,
        trials=args.trials,
        start_trial_index=args.start_trial_index,
        on_summary=_print_summary,
        judge_client=judge_client,
        concurrency=concurrency,
        trial_delay_seconds=args.trial_delay_seconds,
        **common,
    )


async def async_main() -> None:
    args = build_parser().parse_args()
    await _run(args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
