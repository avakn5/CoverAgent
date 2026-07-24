from __future__ import annotations

from typing import Protocol

from coveragent.schemas import ChatMessage, GenerationParams, ModelSpec


class ChatClient(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        """Return one assistant completion for a chat message list."""
