from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from sec_agent.llm.cache import ReplayCache
from sec_agent.llm.client import (
    LLMClient,
    LLMRefusalError,
    LLMTruncatedError,
    Role,
)
from sec_agent.llm.ledger import BudgetExceededError, CostLedger, LedgerEntry
from sec_agent.llm.pricing import UnknownModelPriceError, cost_usd, worst_case_cost_usd


# The structured output every fake call returns. No test here makes a real API call.
class Answer(BaseModel):
    value: str


def response(
    *,
    output: BaseModel | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 1_000,
    output_tokens: int = 100,
    category: str | None = None,
) -> SimpleNamespace:
    """A stand-in for the SDK's parsed response, with only the fields the client reads."""
    return SimpleNamespace(
        parsed_output=output,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class FakeAnthropic:
    """Plays back queued responses and records each request. An Exception in the queue is raised."""

    def __init__(self, *responses: SimpleNamespace) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    async def _parse(self, **request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(settings, fake, *, max_usd=1.0, cache=None):
    """An LLMClient wired to the fake, with its ledger returned so tests can read spend."""
    ledger = CostLedger(max_usd)
    return LLMClient(settings, ledger, client=fake, cache=cache), ledger


def test_cost_uses_model_rates():
    assert cost_usd("claude-opus-5-5", input_tokens=1_000_000, output_tokens=0) == 4.0
    assert cost_usd("claude-haiku-5-5", input_tokens=0, output_tokens=1_000_000) == 0.5


def test_haiku_long_prompt_uses_higher_rates():
    # 200,000 input tokens is over the 100,000 threshold: $0.50 per million, not $0.10.
    assert cost_usd("claude-haiku-5-5", input_tokens=200_000, output_tokens=0) == pytest.approx(0.1)


def test_cache_tokens_are_priced():
    # Reads cost $0.20 per million for Opus. Writes cost 1.25 times the $4.00 input rate.
    assert cost_usd(
        "claude-opus-5-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
    ) == pytest.approx(0.2)
    assert cost_usd(
        "claude-opus-5-5", input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000
    ) == pytest.approx(5.0)


def test_unknown_model_price_raises():
    with pytest.raises(UnknownModelPriceError):
        cost_usd("claude-unknown", input_tokens=1, output_tokens=1)


async def test_returns_parsed_output_and_records_cost(settings):
    fake = FakeAnthropic(response(output=Answer(value="ok")))
    client, ledger = make_client(settings, fake)
    result = await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    assert result.output == Answer(value="ok")
    assert not result.replayed
    expected = cost_usd("claude-haiku-5-5", input_tokens=1_000, output_tokens=100)
    assert ledger.spent_usd == pytest.approx(expected)
    assert ledger.cost_by_role() == {"critic": pytest.approx(expected)}


async def test_roles_use_the_agreed_models(settings):
    client, _ = make_client(settings, FakeAnthropic())
    for role in (Role.MATERIALITY, Role.ENTAILMENT, Role.BRIEF):
        assert client.model_for(role) == "claude-opus-5-5"
    for role in (Role.GENERATOR, Role.CRITIC, Role.RETRIEVAL):
        assert client.model_for(role) == "claude-haiku-5-5"


async def test_fallback_sent_for_opus_but_not_haiku(settings):
    fake = FakeAnthropic(response(output=Answer(value="a")), response(output=Answer(value="b")))
    client, _ = make_client(settings, fake)
    await client.complete(Role.BRIEF, system="s", user="u", output_type=Answer)
    await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    opus, haiku = fake.requests
    assert opus["fallbacks"] == "default"
    assert opus["betas"] == ["server-side-fallback-2026-07-01"]
    assert "fallbacks" not in haiku
    assert "betas" not in haiku


async def test_request_avoids_parameters_current_models_reject(settings):
    # These return a 400 error on current models, so they must never be sent.
    fake = FakeAnthropic(response(output=Answer(value="a")))
    client, _ = make_client(settings, fake)
    await client.complete(Role.BRIEF, system="s", user="u", output_type=Answer)
    request = fake.requests[0]
    for rejected in ("temperature", "top_p", "top_k", "tool_choice", "thinking"):
        assert rejected not in request


async def test_only_system_prompt_is_cached(settings):
    # The stable system prompt is worth caching; each filing passage is different every call.
    fake = FakeAnthropic(response(output=Answer(value="a")))
    client, _ = make_client(settings, fake)
    await client.complete(Role.CRITIC, system="stable", user="varying", output_type=Answer)
    request = fake.requests[0]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in request
    assert request["messages"] == [{"role": "user", "content": "varying"}]


async def test_refusal_raises_and_is_still_charged(settings):
    # A refused request still used tokens, so the ledger must show them.
    fake = FakeAnthropic(response(stop_reason="refusal", category="general_harms"))
    client, ledger = make_client(settings, fake)
    with pytest.raises(LLMRefusalError, match="general_harms"):
        await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    assert ledger.spent_usd > 0


async def test_truncation_raises(settings):
    fake = FakeAnthropic(response(stop_reason="max_tokens"))
    client, _ = make_client(settings, fake)
    with pytest.raises(LLMTruncatedError):
        await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)


async def test_budget_stops_call_before_it_is_sent(settings):
    # The cap ($0.0001) is below even the worst-case cost of one Opus call.
    fake = FakeAnthropic(response(output=Answer(value="never")))
    client, ledger = make_client(settings, fake, max_usd=0.0001)
    with pytest.raises(BudgetExceededError):
        await client.complete(Role.BRIEF, system="s", user="u", output_type=Answer)
    assert fake.requests == []
    assert ledger.spent_usd == 0


async def test_failed_call_releases_its_reservation(settings):
    # The cap fits one call and a half, so a leaked reservation from the failure would block
    # the retry.
    fake = FakeAnthropic(ValueError("network down"), response(output=Answer(value="ok")))
    worst = worst_case_cost_usd("claude-haiku-5-5", estimated_input_tokens=1, max_tokens=2_000)
    client, _ = make_client(settings, fake, max_usd=worst * 1.5)
    with pytest.raises(ValueError):
        await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    result = await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    assert result.output.value == "ok"


def test_in_flight_reservations_count_against_the_cap():
    # Two 0.6 calls can't both be in flight under a cap of 1.0, even though nothing is spent yet.
    ledger = CostLedger(1.0)
    ledger.reserve(0.6)
    with pytest.raises(BudgetExceededError):
        ledger.reserve(0.6)
    ledger.record(LedgerEntry("critic", "m", 1, 1, 0, 0, 0.1, False), reserved=0.6)
    ledger.reserve(0.6)
    assert ledger.spent_usd == pytest.approx(0.1)


def test_ledger_writes_entries_to_file(tmp_path):
    path = tmp_path / "run" / "ledger.jsonl"
    ledger = CostLedger(1.0, path=path)
    ledger.record(LedgerEntry("critic", "m", 1, 1, 0, 0, 0.1, False))
    assert '"role": "critic"' in path.read_text()


async def test_repeat_call_replays_for_free(settings, tmp_path):
    fake = FakeAnthropic(response(output=Answer(value="saved")))
    client, ledger = make_client(settings, fake, cache=ReplayCache(tmp_path / "cache.sqlite"))
    first = await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    spent = ledger.spent_usd
    second = await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    assert second.output == first.output
    assert second.replayed
    assert second.cost_usd == 0
    assert ledger.spent_usd == spent
    assert len(fake.requests) == 1


async def test_changed_prompt_is_not_replayed(settings, tmp_path):
    fake = FakeAnthropic(response(output=Answer(value="a")), response(output=Answer(value="b")))
    client, _ = make_client(settings, fake, cache=ReplayCache(tmp_path / "cache.sqlite"))
    await client.complete(Role.CRITIC, system="s", user="one", output_type=Answer)
    result = await client.complete(Role.CRITIC, system="s", user="two", output_type=Answer)
    assert result.output.value == "b"


async def test_failures_are_not_cached(settings, tmp_path):
    # Caching a refusal would replay it forever; only complete answers are saved.
    fake = FakeAnthropic(response(stop_reason="refusal"), response(output=Answer(value="ok")))
    client, _ = make_client(settings, fake, cache=ReplayCache(tmp_path / "cache.sqlite"))
    with pytest.raises(LLMRefusalError):
        await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    result = await client.complete(Role.CRITIC, system="s", user="u", output_type=Answer)
    assert result.output.value == "ok"


async def test_runnable_composes_in_lcel(settings):
    # The Tree-of-Thought generator and critic are LCEL chains, so a call must pipe like one.
    fake = FakeAnthropic(response(output=Answer(value="chained")))
    client, _ = make_client(settings, fake)
    chain = RunnableLambda(lambda text: f"Question: {text}") | client.runnable(
        Role.GENERATOR, system="s", output_type=Answer
    )
    assert await chain.ainvoke("why") == Answer(value="chained")
    assert fake.requests[0]["messages"][0]["content"] == "Question: why"
