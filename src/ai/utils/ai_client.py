"""Thin wrapper around the Anthropic SDK for calling Claude models."""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


def call_haiku(prompt: str, system: str = "", max_tokens: int = 512) -> str:
    """Call claude-haiku-4-5-20251001. Returns the text response."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system or "You are a crypto trading signal generator. Respond only with valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_model(model: str, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Call any Claude model by ID. Returns the text response."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system or "You are a trading system analyst. Respond only with valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
