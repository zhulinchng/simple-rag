from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel


def call_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    llm_cfg: dict,
) -> BaseModel:
    """Dispatch to the configured LLM provider and return a validated Pydantic object."""
    provider = llm_cfg.get("provider", "openai")
    model = llm_cfg["model"]
    temperature = llm_cfg.get("temperature", 0.0)
    max_tokens = llm_cfg.get("max_tokens", 1024)
    base_url = llm_cfg.get("base_url") or None  # coerce null/empty to None

    if provider == "openai":
        return _call_openai(
            system_prompt,
            user_prompt,
            response_model,
            model,
            temperature,
            max_tokens,
            base_url,
        )
    elif provider == "local":
        if not base_url:
            raise ValueError("llm.base_url must be set when provider is 'local'")
        return _call_local(system_prompt, user_prompt, response_model, model, base_url)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Use 'openai' or 'local'."
        )


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
) -> BaseModel:
    import openai

    client = openai.OpenAI(base_url=base_url)
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.parsed


def _call_local(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    model: str,
    base_url: str,
) -> BaseModel:
    import requests

    schema = response_model.model_json_schema()
    augmented_system = (
        system_prompt
        + "\n\nRESPOND ONLY WITH VALID JSON that matches this exact schema."
        + " Do not include markdown, code fences, explanation, or any text outside the JSON object.\n"
        + f"Schema:\n{json.dumps(schema, indent=2)}"
    )

    payload = {"model": model, "system_prompt": augmented_system, "input": user_prompt}
    resp = requests.post(f"{base_url}/api/v1/chat", json=payload, timeout=60)
    resp.raise_for_status()

    text = _extract_text(resp.json())
    # Strip markdown code fences if the model adds them despite instructions
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return response_model.model_validate_json(text)


def _extract_text(raw: Any) -> str:
    """Extract the text content from various local LLM response shapes."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # Shape: {"output": [{"type": "message", "content": "..."}], ...}
        output = raw.get("output")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    return item["content"]
        # Shape: {"output": "...", ...} or {"response": "...", ...}
        for key in ("output", "response", "text", "content"):
            if key in raw and isinstance(raw[key], str):
                return raw[key]
        # OpenAI-compatible fallback
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
    raise ValueError(f"Cannot extract text from local LLM response: {raw!r}")
