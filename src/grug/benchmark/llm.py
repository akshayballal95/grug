"""Thin LLM client for the benchmark, via litellm.

litellm is used rather than a vendor SDK so the same benchmark runs against
Bedrock, OpenAI, Anthropic or a local model by changing one string. The
provider prefix is litellm's: ``bedrock/...``, ``openai/...``, and so on.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

__all__ = ["ANSWER_PROMPT", "LLMClient", "StubClient"]

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
class LLMClient:
    """Calls a chat model through litellm, with retries and a thread pool."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 64
    workers: int = 8
    retries: int = 3
    timeout: int = 120

    def __post_init__(self) -> None:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - guarded by the extra
            raise ImportError(
                "grug.benchmark needs 'litellm'. Install with: pip install 'grug[bench]'"
            ) from exc
        litellm.drop_params = True
        self._litellm = litellm

    def one(self, context: str, question: str) -> str:
        """Answer a single question, returning "" if every attempt fails."""
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        for attempt in range(self.retries):
            try:
                response = self._litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                if attempt == self.retries - 1:
                    print(f"    llm error after {self.retries} tries: {str(exc)[:120]}", flush=True)
                    return ""
                time.sleep(2**attempt)
        return ""

    def many(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Answer (context, question) pairs concurrently, preserving order."""
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda p: self.one(*p), pairs))


@dataclass
class StubClient:
    """Offline stand-in: echoes a slice of the context.

    Lets the benchmark pipeline be exercised in tests without API calls.
    """

    words: int = 4

    def one(self, context: str, question: str) -> str:
        return " ".join(context.split()[: self.words])

    def many(self, pairs: list[tuple[str, str]]) -> list[str]:
        return [self.one(c, q) for c, q in pairs]


def credentials_present() -> dict[str, bool]:
    """Which provider credentials are visible, for a clear error up front."""
    return {
        "aws": bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")),
        "aws_region": bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
