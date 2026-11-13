"""The single path from agents to Claude.

Every call returns a validated Pydantic object, is charged against the run's
budget, and can be replayed. `runnable` exposes a call as an LCEL Runnable so
generator and critic chains compose inside LangGraph nodes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import anthropic
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from sec_agent.config import Settings
from sec_agent.llm.cache import ReplayCache, cache_key
from sec_agent.llm.ledger import CostLedger, LedgerEntry
from sec_agent.llm.pricing import cost_usd, get_price, worst_case_cost_usd

# How much thinking the model does per call. Higher costs more and takes longer.
Effort = Literal["low", "medium", "high", "xhigh", "max"]

CHARS_PER_TOKEN_ESTIMATE = 2  # deliberately low, so budget reservations err high
# Beta flag that lets the API retry a refused request on another model by itself.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Haiku has no server-side fallback: the API documents this for claude-haiku-5-5.
MODELS_WITH_SERVER_FALLBACK = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})


class Role(StrEnum):
    """The jobs the system gives Claude. Each has a model tier and limits in ROLE_CONFIG."""

    MATERIALITY = "materiality"
    ENTAILMENT = "entailment"
    BRIEF = "brief"
    GENERATOR = "generator"
    CRITIC = "critic"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class RoleConfig:
    # "strong" is the expensive model, used for the few high-stakes calls; "worker" is the
    # cheap model, used for the many calls inside Tree-of-Thought.
    tier: Literal["strong", "worker"]
    effort: Effort
    max_tokens: int


# Per-role settings. max_tokens caps the output and sets the worst-case cost reserved against
# the budget, so it is kept near what each role needs rather than set high. Chosen by hand;
# the planned cost measurements will replace them with real numbers.
ROLE_CONFIG: dict[Role, RoleConfig] = {
    Role.MATERIALITY: RoleConfig("strong", "medium", 4_000),
    Role.ENTAILMENT: RoleConfig("strong", "high", 4_000),
    # Was 16,000, which reserved a $0.33 worst case per summary call against the run cap. The one
    # live summary (2026-11-10, two findings) used 516 output tokens, thinking included, so 4,000
    # leaves room for a longer brief; a cut-off answer falls back to the template summary.
    Role.BRIEF: RoleConfig("strong", "high", 4_000),
    Role.GENERATOR: RoleConfig("worker", "low", 4_000),
    Role.CRITIC: RoleConfig("worker", "low", 2_000),
    Role.RETRIEVAL: RoleConfig("worker", "low", 2_000),
}


class LLMError(RuntimeError):
    """An LLM call finished without a usable result."""


class LLMRefusalError(LLMError):
    """The model declined to answer. Callers record this and send the item to human review."""

    def __init__(self, model: str, category: str | None) -> None:
        super().__init__(f"{model} declined the request (category: {category or 'unspecified'}).")
        self.model = model
        self.category = category


class LLMTruncatedError(LLMError):
    """The response hit max_tokens before the structured output was complete."""


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    """A parsed answer with what it cost. `replayed` means it came from the cache."""

    output: T
    model: str
    cost_usd: float
    replayed: bool


@dataclass(frozen=True)
class PlannedCall:
    """A request looked at before it is sent (`LLMClient.plan`)."""

    # The replay cache holds its answer, so `complete` would cost nothing and reserve nothing.
    cached: bool
    # What `complete` would reserve against the ledger if it is not cached.
    worst_case_usd: float


class LLMClient:
    """Sends requests to Claude, charges the ledger, and uses the replay cache if given."""

    def __init__(
        self,
        settings: Settings,
        ledger: CostLedger,
        *,
        cache: ReplayCache | None = None,
        client: Any | None = None,
        max_concurrency: int = 5,
    ) -> None:
        self._settings = settings
        self._ledger = ledger
        self._cache = cache
        self._client = client or anthropic.AsyncAnthropic(
            api_key=settings.require_api_key(), max_retries=4
        )
        # Limits how many requests are in flight at once.
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def model_for(self, role: Role) -> str:
        """The model name a role currently uses, from settings."""
        tier = ROLE_CONFIG[role].tier
        return self._settings.model_strong if tier == "strong" else self._settings.model_worker

    async def complete[T: BaseModel](
        self,
        role: Role,
        *,
        system: str,
        user: str,
        output_type: type[T],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[T]:
        model, effort, max_tokens = self._settle(role, effort, max_tokens)
        key = self._key(model, system, user, output_type, effort, max_tokens)
        # A cache hit costs nothing but is still logged, so the ledger shows every call.
        if self._cache is not None and (payload := self._cache.get(key)) is not None:
            output = output_type.model_validate_json(payload)
            self._ledger.record(LedgerEntry(role.value, model, 0, 0, 0, 0, 0.0, replayed=True))
            return LLMResult(output=output, model=model, cost_usd=0.0, replayed=True)

        # Hold the worst-case cost before sending, so parallel calls can't overshoot the budget.
        reserved = self._worst_case(model, system, user, max_tokens)
        self._ledger.reserve(reserved)
        recorded = False
        try:
            # The semaphore is taken after reserving, so queued calls still count against the cap.
            async with self._semaphore:
                response = await self._client.beta.messages.parse(
                    **self._request(model, system, user, output_type, effort, max_tokens)
                )
            usage = response.usage
            cost = cost_usd(
                model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens or 0,
                cache_write_tokens=usage.cache_creation_input_tokens or 0,
            )
            # Failed responses are still billed, so record usage before checking them.
            self._ledger.record(
                LedgerEntry(
                    role=role.value,
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_input_tokens or 0,
                    cache_write_tokens=usage.cache_creation_input_tokens or 0,
                    cost_usd=cost,
                    replayed=False,
                    fallback_used=_used_fallback(response),
                ),
                reserved=reserved,
            )
            recorded = True
        finally:
            # If the request itself failed, no money was spent, so give the reservation back.
            if not recorded:
                self._ledger.release(reserved)

        # Only complete, parsed answers are returned or cached.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(model, getattr(details, "category", None))
        if response.stop_reason == "max_tokens":
            raise LLMTruncatedError(f"{model} hit max_tokens={max_tokens} for role {role.value}.")
        output = response.parsed_output
        if output is None:
            raise LLMError(f"{model} returned no parsable {output_type.__name__}.")

        if self._cache is not None:
            self._cache.put(key, output.model_dump_json())
        return LLMResult(output=output, model=model, cost_usd=cost, replayed=False)

    def plan[T: BaseModel](
        self,
        role: Role,
        *,
        system: str,
        user: str,
        output_type: type[T],
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> PlannedCall:
        """What `complete` would do with this request, without sending it or touching the ledger.

        For checks before a run spends anything: a step whose answers are all cached is free.
        It uses the same key and the same worst case as `complete`, so the two cannot disagree.
        """
        model, effort, max_tokens = self._settle(role, effort, max_tokens)
        key = self._key(model, system, user, output_type, effort, max_tokens)
        return PlannedCall(
            cached=self._cache is not None and self._cache.get(key) is not None,
            worst_case_usd=self._worst_case(model, system, user, max_tokens),
        )

    def _settle(
        self, role: Role, effort: Effort | None, max_tokens: int | None
    ) -> tuple[str, Effort, int]:
        config = ROLE_CONFIG[role]
        model = self.model_for(role)
        get_price(model)  # fail before sending if this model's spend can't be tracked
        return model, effort or config.effort, max_tokens or config.max_tokens

    @staticmethod
    def _key(
        model: str,
        system: str,
        user: str,
        output_type: type[BaseModel],
        effort: Effort,
        max_tokens: int,
    ) -> str:
        # Everything that changes the answer is part of the key.
        return cache_key(
            model=model,
            system=system,
            user=user,
            schema=output_type.model_json_schema(),
            effort=effort,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _worst_case(model: str, system: str, user: str, max_tokens: int) -> float:
        estimated_input = (len(system) + len(user)) // CHARS_PER_TOKEN_ESTIMATE
        return worst_case_cost_usd(
            model, estimated_input_tokens=estimated_input, max_tokens=max_tokens
        )

    def runnable[T: BaseModel](
        self,
        role: Role,
        *,
        system: str,
        output_type: type[T],
        effort: Effort | None = None,
    ) -> RunnableLambda:
        """An LCEL step that takes the user prompt as a string and returns the parsed output."""

        async def call(user: str) -> T:
            result = await self.complete(
                role, system=system, user=user, output_type=output_type, effort=effort
            )
            return result.output

        return RunnableLambda(_sync_not_supported(role), afunc=call, name=f"claude_{role.value}")

    @staticmethod
    def _request(
        model: str,
        system: str,
        user: str,
        output_type: type[BaseModel],
        effort: Effort,
        max_tokens: int,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # Cache only the stable system prompt. Caching the whole request
            # would pay the write premium for every unique filing passage.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_format": output_type,
            "output_config": {"effort": effort},
        }
        if model in MODELS_WITH_SERVER_FALLBACK:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request


def _used_fallback(response: Any) -> bool:
    """True if the API answered with its fallback model after the first one refused."""
    iterations = getattr(response.usage, "iterations", None) or []
    return any(getattr(entry, "type", None) == "fallback_message" for entry in iterations)


def _sync_not_supported(role: Role) -> Callable[[str], Any]:
    # LCEL needs a sync function as well as an async one; calls here are async-only.
    def call(_: str) -> Any:
        raise NotImplementedError(f"The {role.value} step is async. Use ainvoke.")

    return call
