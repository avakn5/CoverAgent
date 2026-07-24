from __future__ import annotations

from dataclasses import dataclass

from coveragent.clients import ChatClient
from coveragent.schemas import (
    ChatMessage,
    GenerationParams,
    ModelSpec,
    Speaker,
    TranscriptEvent,
)


@dataclass
class BaseAgent:
    name: Speaker
    system_prompt: str
    model: ModelSpec
    generation: GenerationParams
    client: ChatClient

    async def respond(
        self,
        transcript: list[TranscriptEvent],
        instruction: str | None = None,
        generation: GenerationParams | None = None,
    ) -> str:
        messages = self._build_messages(transcript, instruction)
        return await self.client.complete(messages, self.model, generation or self.generation)

    def _build_messages(
        self,
        transcript: list[TranscriptEvent],
        instruction: str | None,
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        for event in transcript:
            if event.speaker == self.name:
                messages.append(ChatMessage(role="assistant", content=event.content))
            else:
                # Attribute the peer's turns by name so the model never mistakes
                # the other agent's words (or requests) for instructions from its
                # own principal.
                messages.append(
                    ChatMessage(role="user", content=f"{event.speaker}: {event.content}")
                )

        if instruction:
            messages.append(ChatMessage(role="user", content=instruction))

        return messages
