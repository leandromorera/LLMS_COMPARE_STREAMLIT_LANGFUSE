from __future__ import annotations

import os

from dataclasses import dataclass



@dataclass(frozen=True)

class Settings:

    openai_api_key: str

    langfuse_public_key: str | None

    langfuse_secret_key: str | None

    langfuse_host: str | None

    judge_models: tuple[str, str, str]

    openai_timeout_seconds: int



def load_settings() -> Settings:

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not openai_api_key:

        raise RuntimeError("OPENAI_API_KEY is required (set it in your environment or .env).")



    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    langfuse_host = os.getenv("LANGFUSE_HOST")



    judge_models = (

        os.getenv("JUDGE_MODEL_1", "gpt-4o-mini"),

        os.getenv("JUDGE_MODEL_2", "gpt-4.1-mini"),

        os.getenv("JUDGE_MODEL_3", "gpt-4o"),

    )



    openai_timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))



    return Settings(

        openai_api_key=openai_api_key,

        langfuse_public_key=langfuse_public_key,

        langfuse_secret_key=langfuse_secret_key,

        langfuse_host=langfuse_host,

        judge_models=judge_models,

        openai_timeout_seconds=openai_timeout_seconds,

    )
