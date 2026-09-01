"""Approximate model token prices, used only to fill the `est_cost_usd` field in
the event log — this is not billing. Verify current numbers on the provider page;
Moonshot's official Kimi rates and third-party (OpenRouter/DeepInfra) rates differ.
"""

from __future__ import annotations

# model id -> (USD per 1M input tokens, USD per 1M output tokens)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "kimi-k2.6": (0.55, 2.65),
    "kimi-k2.7-code": (0.55, 2.65),
    "kimi-k3": (3.00, 15.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough USD cost of one call. Unknown model -> 0.0 (logged, not charged)."""
    rate = MODEL_PRICES.get(model)
    if rate is None:
        return 0.0
    return round(input_tokens / 1_000_000 * rate[0] + output_tokens / 1_000_000 * rate[1], 6)
