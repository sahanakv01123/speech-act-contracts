import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AzureOpenAIConfig:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str


def load_dotenv_if_present() -> None:
    root = Path(__file__).resolve().parents[3]
    dotenv_path = root / ".env"
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


def load_azure_openai_config() -> AzureOpenAIConfig:
    load_dotenv_if_present()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Azure OpenAI environment variables: {', '.join(missing)}")

    return AzureOpenAIConfig(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
    )


def build_chat_completions_url(config: AzureOpenAIConfig) -> str:
    return (
        f"{config.endpoint}/openai/deployments/{config.deployment}/chat/completions"
        f"?api-version={config.api_version}"
    )


def build_azure_openai_client(config: AzureOpenAIConfig):
    try:
        from openai import AzureOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is required for Azure generation. Install it with `pip install openai`."
        ) from exc

    return AzureOpenAI(
        api_key=config.api_key,
        azure_endpoint=config.endpoint,
        api_version=config.api_version,
    )


def generate_chat_completion(
    config: AzureOpenAIConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 180,
) -> str:
    client = build_azure_openai_client(config)
    convo = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=config.deployment,
        messages=convo,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    try:
        return (resp.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Azure OpenAI response shape: {resp}") from exc
