"""Model-agnostic generation dispatch for benchmark scripts.

Provides a single entry point that resolves one of the three evaluated
deployments (``azure`` = GPT-4o, ``claude`` = Claude Sonnet 4.6,
``gpt52`` = GPT-5.2) to a loaded config plus a ``generate(system_prompt,
user_prompt)`` callable, so baseline scripts do not duplicate per-model wiring.
"""

from pathlib import Path
from typing import Callable, Tuple
import os


MODEL_KEYS = ("azure", "claude", "gpt52")

_ROOT = Path(__file__).resolve().parents[3]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_generator(model_key: str) -> Tuple[object, Callable[[str, str], str], str]:
    """Return ``(config, generate_fn, model_name)`` for the requested model.

    ``generate_fn`` has signature ``(system_prompt, user_prompt) -> str`` and
    applies the shared decoding settings (temperature 0.2) used everywhere else.
    """
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model key {model_key!r}; expected one of {MODEL_KEYS}.")

    if model_key == "claude":
        from speech_act_contract.llm.azure_claude import (
            generate_claude_message,
            load_azure_claude_config,
        )

        config = load_azure_claude_config()

        def generate(system_prompt: str, user_prompt: str) -> str:
            return generate_claude_message(
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        return config, generate, config.deployment

    # azure (GPT-4o) and gpt52 both use the Azure OpenAI client but load from
    # different env profiles; gpt52 credentials live in .env.gpt52.
    if model_key == "gpt52":
        _load_env_file(_ROOT / ".env.gpt52")

    from speech_act_contract.llm.azure_openai import (
        generate_chat_completion,
        load_azure_openai_config,
    )

    config = load_azure_openai_config()

    def generate(system_prompt: str, user_prompt: str) -> str:
        return generate_chat_completion(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    return config, generate, config.deployment
