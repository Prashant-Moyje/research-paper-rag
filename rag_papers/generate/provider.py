"""LLM provider abstraction.

Generation sits behind a Protocol so the backend is a configuration choice, not
a code change. The project runs fully locally on Ollama; the same interface
would accept a hosted API later without touching prompt, citation or evaluation
code.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...
    def health(self) -> tuple[bool, str]: ...


class OllamaProvider:
    """Local generation through the Ollama HTTP API.

    Endpoint note: Ollama exposes both /api/generate and /api/chat, and which
    one a model accepts varies by build. We try chat first and fall back to
    generate, so the project is not pinned to one server version.
    """

    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 300,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # -- internals ---------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        import requests

        resp = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"{path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _options(self) -> dict:
        return {"temperature": self.temperature, "num_predict": self.max_tokens}

    # -- public ------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        """Check the server is reachable and the model is present."""
        import requests

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            return False, (
                f"Cannot reach Ollama at {self.host} ({exc}). "
                f"Start it with:  ollama serve"
            )

        names = {m.get("name", "") for m in resp.json().get("models", [])}
        if self.model not in names:
            return False, (
                f"Model {self.model!r} is not installed. Pull it with:\n"
                f"    ollama pull {self.model}\n"
                f"Installed: {', '.join(sorted(names)) or '(none)'}"
            )
        return True, f"Ollama OK - {self.model}"

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]

        data: dict | None = None
        errors: list[str] = []

        try:
            data = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": self._options(),
                },
            )
            text = data.get("message", {}).get("content", "")
        except RuntimeError as exc:
            errors.append(str(exc))
            try:
                joined = (f"{system}\n\n" if system else "") + prompt
                data = self._post(
                    "/api/generate",
                    {
                        "model": self.model,
                        "prompt": joined,
                        "stream": False,
                        "options": self._options(),
                    },
                )
                text = data.get("response", "")
            except RuntimeError as exc2:
                errors.append(str(exc2))
                raise RuntimeError(
                    "Ollama rejected both /api/chat and /api/generate:\n  "
                    + "\n  ".join(errors)
                    + "\n\nIf the model was just pulled, restart the Ollama server - "
                    "it can report 'does not support chat/generate' until it is restarted."
                ) from exc2

        return LLMResponse(
            text=(text or "").strip(),
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_s=time.perf_counter() - t0,
        )


def get_provider(cfg) -> LLMProvider:
    """Factory: resolve RAG_LLM_BACKEND to a provider."""
    backend = (cfg.llm_backend or "ollama").lower()
    if backend == "ollama":
        return OllamaProvider(
            host=cfg.ollama_host,
            model=cfg.ollama_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.llm_timeout,
        )
    raise ValueError(
        f"Unknown RAG_LLM_BACKEND={backend!r}. Supported: 'ollama'."
    )
