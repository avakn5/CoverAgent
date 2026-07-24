"""Minimal Fireworks client configuration path for the agent-workforce service."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import FIREWORKS_API_KEY, FIREWORKS_BASE_URL, REQUEST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class FireworksClientConfig:
    base_url: str
    api_key_suffix: str
    timeout_seconds: int


def build_client_config() -> FireworksClientConfig:
    """Return the client settings used by worker startup.

    Only the last four characters of the key are surfaced as a suffix, so the
    full credential is never passed around or printed.
    """
    return FireworksClientConfig(
        base_url=FIREWORKS_BASE_URL,
        api_key_suffix=FIREWORKS_API_KEY[-4:],
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )

