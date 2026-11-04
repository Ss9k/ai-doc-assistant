from decimal import Decimal

# USD per 1 million tokens (paid Standard tier).
PRICING = {
    "gemini": {
        "gemini-2.5-flash": {
            "input": Decimal("0.30"),
            "output": Decimal("2.50"),
        },
        "gemini-3.5-flash-lite": {
            "input": Decimal("0.075"),
            "output": Decimal("0.30"),
        },
        "gemini-3.8-flash": {
            "input": Decimal("0.075"),
            "output": Decimal("0.30"),
        },
    },
    "ollama": {
        "qwen3:8b": {
            "input": Decimal("0"),
            "output": Decimal("0"),
        },
        "llama3.2": {
            "input": Decimal("0"),
            "output": Decimal("0"),
        },
    },
}


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    provider_key = provider.lower().strip()
    
    # Fallback default pricing if model is not found
    default_rates = {"input": Decimal("0"), "output": Decimal("0")}
    
    rates = PRICING.get(provider_key, {}).get(model, default_rates)

    million = Decimal("1000000")

    cost = (
        Decimal(input_tokens) * rates["input"]
        + Decimal(output_tokens) * rates["output"]
    ) / million

    return cost.quantize(Decimal("0.00000001"))