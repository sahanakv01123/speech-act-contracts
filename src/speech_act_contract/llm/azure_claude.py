import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AzureClaudeConfig:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str


def load_dotenv_profile_if_present(profile_name: str) -> None:
    root = Path(__file__).resolve().parents[3]
    dotenv_path = root / profile_name
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_azure_claude_config() -> AzureClaudeConfig:
    load_dotenv_profile_if_present(".env.claude")

    api_key = os.environ.get("AZURE_ANTHROPIC_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_ANTHROPIC_ENDPOINT", "").strip().rstrip("/")
    deployment = os.environ.get("AZURE_ANTHROPIC_DEPLOYMENT", "").strip()
    api_version = os.environ.get("AZURE_ANTHROPIC_API_VERSION", "").strip()

    missing = [
        name
        for name, value in (
            ("AZURE_ANTHROPIC_API_KEY", api_key),
            ("AZURE_ANTHROPIC_ENDPOINT", endpoint),
            ("AZURE_ANTHROPIC_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Azure Claude environment variables: {', '.join(missing)}")

    return AzureClaudeConfig(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
    )


def build_azure_claude_client(config: AzureClaudeConfig):
    try:
        from anthropic import AnthropicFoundry
    except ImportError as exc:
        raise RuntimeError(
            "The `anthropic` package is required for Claude generation. Install it with `pip install anthropic`."
        ) from exc

    return AnthropicFoundry(
        api_key=config.api_key,
        base_url=config.endpoint,
    )


def flatten_message_content(message_content) -> str:
    if isinstance(message_content, str):
        return message_content.strip()

    if isinstance(message_content, list):
        texts = []
        for block in message_content:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts).strip()

    return str(message_content).strip()


def generate_claude_message(
    config: AzureClaudeConfig,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 180,
    temperature: float = 0.2,
) -> str:
    client = build_azure_claude_client(config)

    message = client.messages.create(
        model=config.deployment,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return flatten_message_content(message.content)
