"""Claude API prices, used to enforce the per-run budget.

An unknown model raises instead of defaulting to zero, so a model change can't
silently disable the budget. Prices are USD per million tokens and should be
checked against Anthropic's pricing page when a model changes.
"""

from __future__ import annotations

from dataclasses import dataclass

CACHE_WRITE_MULTIPLIER = 1.25  # default 5-minute TTL
# Haiku charges higher rates once a prompt passes 100,000 tokens (see PRICES below).
HAIKU_LONG_PROMPT_THRESHOLD = 100_000


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per million tokens. Long-prompt fields apply above the threshold."""

    input: float
    output: float
    cache_read: float
    long_prompt_threshold: int | None = None
    long_prompt_input: float | None = None
    long_prompt_output: float | None = None


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-5-5": ModelPrice(
        input=0.10,
        output=0.50,
        cache_read=0.01,
        long_prompt_threshold=HAIKU_LONG_PROMPT_THRESHOLD,
        long_prompt_input=0.50,
        long_prompt_output=2.50,
    ),
}


class UnknownModelPriceError(KeyError):
    """No price is recorded for this model, so its spend can't be tracked."""


def get_price(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError:
        raise UnknownModelPriceError(
            f"No price recorded for {model!r}. Add it to sec_agent/llm/pricing.py."
        ) from None


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    price = get_price(model)
    input_rate, output_rate = price.input, price.output
    # Long-prompt pricing depends on the whole prompt, cached or not.
    total_input = input_tokens + cache_read_tokens + cache_write_tokens
    if price.long_prompt_threshold is not None and total_input > price.long_prompt_threshold:
        input_rate = price.long_prompt_input or input_rate
        output_rate = price.long_prompt_output or output_rate
    return (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000


def worst_case_cost_usd(model: str, *, estimated_input_tokens: int, max_tokens: int) -> float:
    """Upper bound for one uncached call, used to reserve budget before sending it."""
    # Assume the most expensive case: all input written to the cache and the maximum output.
    return cost_usd(
        model,
        input_tokens=0,
        output_tokens=max_tokens,
        cache_write_tokens=estimated_input_tokens,
    )
