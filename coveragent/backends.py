"""Local inference backends implementing the ChatClient protocol for HF open source models.
"""

from __future__ import annotations

from dataclasses import asdict

from coveragent.models import requires_trust_remote_code, supports_qwen_thinking
from coveragent.schemas import ChatMessage, GenerationParams, ModelSpec


def _env_enabled(name: str, default: bool = True) -> bool:
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _cacheable_api_messages(
    messages: list[ChatMessage],
    *,
    enable_cache: bool,
) -> list[dict]:
    """Render OpenAI-compatible messages, optionally adding prompt-cache hints.

    Fireworks/OpenRouter accept Anthropic-style cache-control text blocks for
    providers that support explicit prompt caching. We cache the stable system
    prompt and, on multi-turn calls, the transcript prefix immediately before the
    current instruction. The final instruction remains uncached because it is the
    changing suffix for this request.
    """
    if not enable_cache or not messages:
        return [{"role": m.role, "content": m.content} for m in messages]

    cache_indices = {0}
    if len(messages) >= 3:
        cache_indices.add(len(messages) - 2)

    rendered: list[dict] = []
    for index, message in enumerate(messages):
        content: object = message.content
        if index in cache_indices and message.content.strip():
            content = [
                {
                    "type": "text",
                    "text": message.content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        rendered.append({"role": message.role, "content": content})
    return rendered


def _collapsed_single_user_turn(chat_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system_parts: list[str] = []
    body_parts: list[str] = []

    for message in chat_messages:
        content = message["content"].strip()
        if not content:
            continue
        if message["role"] == "system":
            system_parts.append(content)
        elif message["role"] == "assistant":
            body_parts.append(f"Assistant previous message:\n{content}")
        else:
            body_parts.append(f"User or other participant:\n{content}")

    collapsed: list[dict[str, str]] = []
    if system_parts:
        collapsed.append({"role": "system", "content": "\n\n".join(system_parts)})
    collapsed.append(
        {
            "role": "user",
            "content": (
                "Use the conversation history below and respond only with the next "
                "assistant message requested by the final instruction.\n\n"
                + "\n\n".join(body_parts)
            ).strip(),
        }
    )
    return collapsed


def _manual_prompt(chat_messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in chat_messages:
        role = message["role"].upper()
        lines.append(f"{role}:\n{message['content'].strip()}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _missing_chat_template(error: Exception) -> bool:
    return "chat_template is not set" in str(error)


def _apply_chat_template(tokenizer, chat_messages: list[dict[str, str]], model_id: str, params: GenerationParams) -> str:
    chat_template_kwargs = {
        "conversation": chat_messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if supports_qwen_thinking(model_id):
        chat_template_kwargs["enable_thinking"] = (
            params.enable_thinking if params.enable_thinking is not None else True
        )
    try:
        return tokenizer.apply_chat_template(**chat_template_kwargs)
    except TypeError:
        chat_template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(**chat_template_kwargs)


def _render_prompt(tokenizer, chat_messages: list[dict[str, str]], model_id: str, params: GenerationParams) -> str:
    try:
        return _apply_chat_template(tokenizer, chat_messages, model_id, params)
    except Exception as original_error:
        collapsed_messages = _collapsed_single_user_turn(chat_messages)
        if collapsed_messages != chat_messages:
            try:
                return _apply_chat_template(tokenizer, collapsed_messages, model_id, params)
            except Exception as collapsed_error:
                if _missing_chat_template(original_error) or _missing_chat_template(collapsed_error):
                    return _manual_prompt(chat_messages)
                raise original_error
        if _missing_chat_template(original_error):
            return _manual_prompt(chat_messages)
        raise


class TransformersChatClient:
    """Hugging Face Transformers backend (one process, models cached by id)."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}

    def _load(self, model_id: str):
        if model_id not in self._models:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            trust_remote_code = requires_trust_remote_code(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
            model.eval()
            self._tokenizers[model_id] = tokenizer
            self._models[model_id] = model
        return self._models[model_id], self._tokenizers[model_id]

    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        import torch

        hf_model, tokenizer = self._load(model.model_id)
        chat_messages = [asdict(message) for message in messages]
        prompt = _render_prompt(tokenizer, chat_messages, model.model_id, params)
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        do_sample = params.temperature > 0
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        with torch.no_grad():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else None,
                top_p=params.top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def _int_env(name: str) -> int | None:
    import os

    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _float_env(name: str) -> float | None:
    import os

    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return float(value)


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    import inspect

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }


class VllmChatClient:
    """vLLM backend for high-throughput local inference.

    This client is meant for one warm Modal worker serving many independent
    trials of the same local model. A short async queue batches concurrent
    ``complete`` calls through vLLM, while the experiment runner preserves each
    dialogue's sequential turn order.
    """

    def __init__(
        self,
        *,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        max_batch_size: int | None = None,
        batch_window_seconds: float = 0.03,
        enable_prefix_caching: bool | None = None,
        dtype: str = "bfloat16",
        enforce_eager: bool | None = None,
        allow_multiple_models: bool = False,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_batch_size = max_batch_size
        self.batch_window_seconds = batch_window_seconds
        self.enable_prefix_caching = enable_prefix_caching
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.allow_multiple_models = allow_multiple_models
        self._engines: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}
        self._queues: dict[str, object] = {}
        self._batch_tasks: dict[str, object] = {}
        self._load_lock = None

    def _default_tensor_parallel_size(self) -> int:
        env_value = _int_env("COVERAGENT_VLLM_TENSOR_PARALLEL_SIZE")
        if env_value is not None:
            return env_value
        try:
            import torch

            return max(1, torch.cuda.device_count())
        except Exception:
            return 1

    async def _load(self, model_id: str):
        if model_id in self._engines:
            return self._engines[model_id], self._tokenizers[model_id]

        import asyncio

        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        async with self._load_lock:
            if model_id in self._engines:
                return self._engines[model_id], self._tokenizers[model_id]
            if self._engines and not self.allow_multiple_models:
                loaded = next(iter(self._engines))
                raise RuntimeError(
                    "VllmChatClient keeps one local model warm per worker by default. "
                    f"Already loaded {loaded!r}; refusing to load {model_id!r}. "
                    "Use judge_size='none', use the same model as judge, or pass a separate API judge_client."
                )

            from transformers import AutoTokenizer

            from vllm import LLM

            trust_remote_code = requires_trust_remote_code(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            engine_kwargs = _filter_supported_kwargs(
                LLM,
                {
                    "model": model_id,
                    "tokenizer": model_id,
                    "trust_remote_code": trust_remote_code,
                    "dtype": self.dtype,
                    "tensor_parallel_size": self.tensor_parallel_size or self._default_tensor_parallel_size(),
                    "max_model_len": self.max_model_len or _int_env("COVERAGENT_VLLM_MAX_MODEL_LEN"),
                    "gpu_memory_utilization": (
                        self.gpu_memory_utilization
                        if self.gpu_memory_utilization is not None
                        else _float_env("COVERAGENT_VLLM_GPU_MEMORY_UTILIZATION")
                    ),
                    "max_num_seqs": self.max_num_seqs or _int_env("COVERAGENT_VLLM_MAX_NUM_SEQS"),
                    "enable_prefix_caching": (
                        self.enable_prefix_caching
                        if self.enable_prefix_caching is not None
                        else _env_enabled("COVERAGENT_VLLM_ENABLE_PREFIX_CACHING", default=True)
                    ),
                    "enforce_eager": self.enforce_eager,
                },
            )
            llm = LLM(**engine_kwargs)
            queue = asyncio.Queue()
            self._engines[model_id] = llm
            self._tokenizers[model_id] = tokenizer
            self._queues[model_id] = queue
            self._batch_tasks[model_id] = asyncio.create_task(self._batch_loop(model_id))
            return llm, tokenizer

    async def warm(self, model_id: str) -> None:
        await self._load(model_id)

    async def _batch_loop(self, model_id: str) -> None:
        import asyncio

        queue = self._queues[model_id]
        llm = self._engines[model_id]
        max_batch_size = self.max_batch_size or self.max_num_seqs or _int_env("COVERAGENT_VLLM_MAX_BATCH_SIZE") or 8
        while True:
            first = await queue.get()
            items = [first]
            deadline = asyncio.get_running_loop().time() + max(0.0, self.batch_window_seconds)
            while len(items) < max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    break

            prompts = [item["prompt"] for item in items]
            sampling_params = [item["sampling_params"] for item in items]
            futures = [item["future"] for item in items]
            try:
                outputs = await asyncio.to_thread(
                    llm.generate,
                    prompts,
                    sampling_params,
                    use_tqdm=False,
                )
                for future, output in zip(futures, outputs):
                    text = output.outputs[0].text.strip() if output.outputs else ""
                    if not future.done():
                        future.set_result(text)
            except Exception as exc:
                for future in futures:
                    if not future.done():
                        future.set_exception(exc)
            finally:
                for _ in items:
                    queue.task_done()

    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        import asyncio

        from vllm import SamplingParams

        _, tokenizer = await self._load(model.model_id)
        chat_messages = [asdict(message) for message in messages]
        prompt = _render_prompt(tokenizer, chat_messages, model.model_id, params)
        sampling_kwargs = _filter_supported_kwargs(
            SamplingParams,
            {
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "seed": params.seed,
                "skip_special_tokens": False,
            },
        )
        sampling_params = SamplingParams(**sampling_kwargs)
        future = asyncio.get_running_loop().create_future()
        await self._queues[model.model_id].put(
            {"prompt": prompt, "sampling_params": sampling_params, "future": future}
        )
        return await future


def _fireworks_extract(data: dict) -> str:
    """Turn a Fireworks chat-completion response into the pipeline's raw string.

    Reasoning models return the chain-of-thought either in a separate
    ``reasoning_content`` field or inline as ``<think>...</think>``. We normalise
    the former into ``<think>...</think>`` tags so the existing
    ``_split_visible_cot`` captures it as ``visible_cot`` with no downstream change.
    """
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if reasoning and "<think>" not in content.lower():
        return f"<think>{reasoning}</think>\n{content}"
    return content


class FireworksChatClient:
    """Fireworks AI backend (OpenAI-compatible /chat/completions).

    Captures reasoning as ``<think>`` tags. Reasoning is left at the model's
    default (on for thinking-capable model ids); pass ``enable_thinking=False``
    (the judge does) to request ``reasoning_effort="none"``. Seed is best-effort
    on Fireworks. Model ids are Fireworks paths (``accounts/fireworks/models/...``).
    """

    BASE_URL = "https://api.fireworks.ai/inference/v1"
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None = None, base_url: str | None = None, max_retries: int = 4) -> None:
        import os

        self._api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set FIREWORKS_API_KEY to use the Fireworks backend.")
        self._base_url = (base_url or os.environ.get("FIREWORKS_BASE_URL") or self.BASE_URL).rstrip("/")
        self._max_retries = max_retries
        self._prompt_cache = _env_enabled("COVERAGENT_PROMPT_CACHE", default=True)
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        return self._client

    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        import asyncio

        import httpx

        payload: dict = {
            "model": model.model_id,
            "messages": _cacheable_api_messages(messages, enable_cache=self._prompt_cache),
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.reasoning_effort is not None:
            payload["reasoning_effort"] = params.reasoning_effort
        elif params.enable_thinking is False:
            payload["reasoning_effort"] = "none"  # best-effort; model-dependent
        elif params.enable_thinking is True:
            payload["reasoning_effort"] = _default_fireworks_reasoning_effort()
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._http().post(url, json=payload, headers=headers)
                if resp.status_code in self._RETRY_STATUS:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}", request=resp.request, response=resp
                    )
                    await asyncio.sleep(min(2 ** attempt, 20))
                    continue
                if self._prompt_cache and resp.status_code in {400, 412, 422}:
                    fallback_payload = dict(payload)
                    fallback_payload["messages"] = _cacheable_api_messages(messages, enable_cache=False)
                    fallback_resp = await self._http().post(url, json=fallback_payload, headers=headers)
                    if fallback_resp.is_error:
                        detail = fallback_resp.text[:1000].strip()
                        message = f"HTTP {fallback_resp.status_code}"
                        if detail:
                            message = f"{message}: {detail}"
                        raise httpx.HTTPStatusError(message, request=fallback_resp.request, response=fallback_resp)
                    return _fireworks_extract(fallback_resp.json())
                resp.raise_for_status()  # non-retryable 4xx (404/401/400) -> raise now
                return _fireworks_extract(resp.json())
            except httpx.TransportError as exc:  # network blip -> retry
                last_error = exc
                await asyncio.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"Fireworks request failed after {self._max_retries} attempts: {last_error}")


def _default_fireworks_reasoning_effort() -> str:
    import os

    return os.environ.get("FIREWORKS_REASONING_EFFORT", "medium")


def _openrouter_extract(data: dict) -> str:
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or msg.get("reasoning_details")
        or ""
    )
    if isinstance(reasoning, list):
        reasoning = "\n".join(
            str(item.get("text") or item.get("content") or item)
            for item in reasoning
        )
    reasoning = str(reasoning).strip()
    if reasoning and "<think>" not in content.lower():
        return f"<think>{reasoning}</think>\n{content}"
    return content


class ModalChatClient:
    """Modal shared endpoint backend (OpenAI-compatible /chat/completions)."""

    BASE_URL = "https://inference.us-west.modal.direct/v1"
    _RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None = None, base_url: str | None = None, max_retries: int | None = None) -> None:
        import os

        self._api_key = api_key or os.environ.get("MODAL_PROXY_TOKEN")
        if not self._api_key:
            raise RuntimeError("Set MODAL_PROXY_TOKEN to use the Modal backend.")
        self._base_url = (base_url or os.environ.get("MODAL_BASE_URL") or self.BASE_URL).rstrip("/")
        self._max_retries = max_retries or int(os.environ.get("MODAL_MAX_RETRIES", "4"))
        self._retry_base_seconds = float(os.environ.get("MODAL_RETRY_BASE_SECONDS", "1"))
        self._retry_max_seconds = float(os.environ.get("MODAL_RETRY_MAX_SECONDS", "20"))
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        return self._client

    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        import asyncio

        import httpx

        payload: dict = {
            "model": model.model_id,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.reasoning_effort is not None:
            payload["reasoning_effort"] = params.reasoning_effort
        elif params.enable_thinking is False:
            payload["reasoning_effort"] = "none"
        elif params.enable_thinking is True:
            payload["reasoning_effort"] = _default_modal_reasoning_effort()

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._http().post(url, json=payload, headers=headers)
                if resp.status_code in self._RETRY_STATUS:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}", request=resp.request, response=resp
                    )
                    await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
                    continue
                resp.raise_for_status()
                return _openrouter_extract(resp.json())
            except httpx.TransportError as exc:
                last_error = exc
                await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
        raise RuntimeError(f"Modal request failed after {self._max_retries} attempts: {last_error}")


def _default_modal_reasoning_effort() -> str:
    import os

    return os.environ.get("MODAL_REASONING_EFFORT", "medium")


class OpenRouterChatClient:
    """OpenRouter backend (OpenAI-compatible /chat/completions)."""

    BASE_URL = "https://openrouter.ai/api/v1"
    _RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None = None, base_url: str | None = None, max_retries: int | None = None) -> None:
        import os

        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY to use the OpenRouter backend.")
        self._base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or self.BASE_URL).rstrip("/")
        self._max_retries = max_retries or int(os.environ.get("OPENROUTER_MAX_RETRIES", "4"))
        self._retry_base_seconds = float(os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "1"))
        self._retry_max_seconds = float(os.environ.get("OPENROUTER_RETRY_MAX_SECONDS", "20"))
        self._prompt_cache = _env_enabled("COVERAGENT_PROMPT_CACHE", default=True)
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        return self._client

    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        import asyncio

        import httpx

        payload: dict = {
            "model": model.model_id,
            "messages": _cacheable_api_messages(messages, enable_cache=self._prompt_cache),
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.enable_thinking is True:
            payload["include_reasoning"] = True
            if params.reasoning_effort is not None:
                payload["reasoning"] = {"effort": str(params.reasoning_effort)}
        elif params.enable_thinking is False:
            # ``include_reasoning=False`` only hides reasoning from OpenRouter's
            # response; it does not stop the model from spending the entire
            # completion budget on hidden reasoning. Explicitly disable it.
            payload["include_reasoning"] = False
            payload["reasoning"] = {"enabled": False}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "CoverAgent",
        }
        url = f"{self._base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._http().post(url, json=payload, headers=headers)
                if resp.status_code in self._RETRY_STATUS:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}", request=resp.request, response=resp
                    )
                    await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
                    continue
                if self._prompt_cache and resp.status_code in {400, 422}:
                    fallback_payload = dict(payload)
                    fallback_payload["messages"] = _cacheable_api_messages(messages, enable_cache=False)
                    fallback_resp = await self._http().post(url, json=fallback_payload, headers=headers)
                    fallback_resp.raise_for_status()
                    return _openrouter_extract(fallback_resp.json())
                resp.raise_for_status()
                try:
                    return _openrouter_extract(resp.json())
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    last_error = exc
                    await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
            except httpx.TransportError as exc:
                last_error = exc
                await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
        raise RuntimeError(f"OpenRouter request failed after {self._max_retries} attempts: {last_error}")


class SailChatClient:
    """Sail Research backend (OpenAI-compatible /chat/completions)."""

    BASE_URL = "https://api.sailresearch.com/v1"
    _RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None = None, base_url: str | None = None, max_retries: int | None = None) -> None:
        import os

        self._api_key = api_key or os.environ.get("SAIL_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set SAIL_API_KEY to use the Sail backend.")
        self._base_url = (base_url or os.environ.get("SAIL_BASE_URL") or self.BASE_URL).rstrip("/")
        self._max_retries = max_retries or int(os.environ.get("SAIL_MAX_RETRIES", "4"))
        self._retry_base_seconds = float(os.environ.get("SAIL_RETRY_BASE_SECONDS", "1"))
        self._retry_max_seconds = float(os.environ.get("SAIL_RETRY_MAX_SECONDS", "20"))
        self._completion_window = os.environ.get("SAIL_COMPLETION_WINDOW", "standard")
        self._send_seed = os.environ.get("SAIL_SEND_SEED", "").lower() in {"1", "true", "yes", "on"}
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0))
        return self._client

    async def complete(
        self,
        messages: list[ChatMessage],
        model: ModelSpec,
        params: GenerationParams,
    ) -> str:
        import asyncio

        import httpx

        payload: dict = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "metadata": {
                "completion_window": self._completion_window,
            },
        }
        if self._send_seed and params.seed is not None:
            payload["seed"] = params.seed
        # For judge : Reasoning models (e.g. GLM-5.2) otherwise spend the whole token budget on
        # hidden thinking and return empty content. The judge sets enable_thinking=False;
        # honor it with reasoning_effort="none" (as the Fireworks client does).
        if params.enable_thinking is False:
            payload["reasoning_effort"] = "none"
        elif params.enable_thinking is True and params.reasoning_effort is not None:
            payload["reasoning_effort"] = str(params.reasoning_effort)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._http().post(url, json=payload, headers=headers)
                if resp.status_code in self._RETRY_STATUS:
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}", request=resp.request, response=resp
                    )
                    await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
                    continue
                if resp.is_error:
                    detail = resp.text[:1000].strip()
                    message = f"HTTP {resp.status_code}"
                    if detail:
                        message = f"{message}: {detail}"
                    raise httpx.HTTPStatusError(message, request=resp.request, response=resp)
                return _openrouter_extract(resp.json())
            except httpx.TransportError as exc:
                last_error = exc
                await asyncio.sleep(min(self._retry_base_seconds * (2 ** attempt), self._retry_max_seconds))
        raise RuntimeError(f"Sail request failed after {self._max_retries} attempts: {last_error}")
