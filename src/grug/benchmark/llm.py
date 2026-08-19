"""Thin LLM client for the benchmark, via litellm.

litellm is used rather than a vendor SDK so the same benchmark runs against
Bedrock, OpenAI, Anthropic or a local model by changing one string. The
provider prefix is litellm's: ``bedrock/...``, ``openai/...``, and so on.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ANSWER_PROMPT", "LLMClient", "StubClient", "Usage"]

#: Deliberately strict: a chatty model scores badly on exact match for reasons
#: that have nothing to do with whether the compression kept the information.
ANSWER_PROMPT = """Answer the question using only the context below.
Reply with the answer alone -- no preamble, no explanation, no full sentence.
If the context does not contain the answer, reply exactly: UNKNOWN

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class Usage:
    """Running totals for one client, so a paid run can report what it spent."""

    calls: int = 0
    failures: int = 0
    truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def summary(self) -> str:
        parts = [
            f"{self.calls} calls",
            f"{self.prompt_tokens / 1000:.0f}k in",
            f"{self.completion_tokens / 1000:.0f}k out",
        ]
        if self.reasoning_tokens:
            share = self.reasoning_tokens / max(1, self.completion_tokens)
            parts.append(f"{self.reasoning_tokens / 1000:.0f}k reasoning ({share:.0%} of out)")
        parts.append(f"${self.cost_usd:.2f}")
        if self.failures:
            parts.append(f"{self.failures} failed")
        if self.truncated:
            parts.append(f"{self.truncated} TRUNCATED")
        return ", ".join(parts)


@dataclass
class LLMClient:
    """Calls a chat model through litellm, with retries and a thread pool."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 64
    workers: int = 8
    retries: int = 3
    timeout: int = 120
    #: Passed through to providers that bill for hidden reasoning tokens.
    #: "low" turns thinking off on Gemini, which costs about a sixth as much.
    #: Note litellm.drop_params silently discards knobs a provider does not
    #: support, so an ineffective setting fails quiet rather than loud.
    reasoning_effort: str | None = None
    #: Called with (completed, total) after each request finishes.
    on_progress: Callable[[int, int], None] | None = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - guarded by the extra
            raise ImportError(
                "grug.benchmark needs 'litellm'. Install with: pip install 'grug[bench]'"
            ) from exc
        litellm.drop_params = True
        self._litellm = litellm
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0

    def _record(self, response: Any, truncated: bool) -> None:
        """Accumulate tokens and cost. Never let accounting break a run."""
        prompt = completion = reasoning = 0
        cost = 0.0
        try:
            u = response.usage
            prompt = getattr(u, "prompt_tokens", 0) or 0
            completion = getattr(u, "completion_tokens", 0) or 0
            details = getattr(u, "completion_tokens_details", None)
            reasoning = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        except Exception:
            pass
        try:
            cost = float(self._litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost = 0.0
        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += prompt
            self.usage.completion_tokens += completion
            self.usage.reasoning_tokens += reasoning
            self.usage.cost_usd += cost
            if truncated:
                self.usage.truncated += 1

    def _tick(self) -> None:
        with self._lock:
            self._done += 1
            done, total = self._done, self._total
        if self.on_progress is not None:
            self.on_progress(done, total)

    def one(self, context: str, question: str) -> str:
        """Answer a single question, returning "" if every attempt fails."""
        return self.complete(ANSWER_PROMPT.format(context=context, question=question))

    def complete(self, prompt: str) -> str:
        """Send a prompt verbatim.

        Distillation needs this: routing a compression instruction through the
        QA template asks the model to answer an empty question *about* the
        instruction, which is not the same task at all.
        """
        for attempt in range(self.retries):
            try:
                extra: dict[str, Any] = {}
                if self.reasoning_effort:
                    extra["reasoning_effort"] = self.reasoning_effort
                response = self._litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    **extra,
                )
                choice = response.choices[0]
                text = (choice.message.content or "").strip()
                # A thinking model can spend the whole budget reasoning and
                # return a half-finished compression. That is worse than an
                # error: it looks like an unusually aggressive one.
                truncated = getattr(choice, "finish_reason", None) == "length"
                self._record(response, truncated)
                return text
            except Exception as exc:
                if attempt == self.retries - 1:
                    print(f"    llm error after {self.retries} tries: {str(exc)[:120]}", flush=True)
                    with self._lock:
                        self.usage.failures += 1
                    return ""
                time.sleep(2**attempt)
        return ""

    def many(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Answer (context, question) pairs concurrently, preserving order."""
        return self._parallel(lambda p: self.one(*p), pairs)

    def complete_many(self, prompts: list[str]) -> list[str]:
        """Send prompts verbatim, concurrently, preserving order."""
        return self._parallel(self.complete, prompts)

    def _parallel(self, fn: Any, items: list[Any]) -> list[str]:
        if not items:
            return []
        with self._lock:
            self._done, self._total = 0, len(items)

        def run(item: Any) -> str:
            try:
                return fn(item)
            finally:
                self._tick()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(run, items))


@dataclass
class StubClient:
    """Offline stand-in: echoes a slice of the context.

    Lets the benchmark pipeline be exercised in tests without API calls.
    """

    words: int = 4
    on_progress: Callable[[int, int], None] | None = None
    usage: Usage = field(default_factory=Usage)

    def one(self, context: str, question: str) -> str:
        return " ".join(context.split()[: self.words])

    def complete(self, prompt: str) -> str:
        return " ".join(prompt.split()[: self.words])

    def many(self, pairs: list[tuple[str, str]]) -> list[str]:
        return [self.one(c, q) for c, q in pairs]

    def complete_many(self, prompts: list[str]) -> list[str]:
        return [self.complete(p) for p in prompts]


def credentials_present() -> dict[str, bool]:
    """Which provider credentials are visible, for a clear error up front."""
    return {
        "aws": bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")),
        "aws_region": bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
