from __future__ import annotations

from dataclasses import dataclass

from typing import Any



@dataclass(frozen=True)

class LangfuseHandle:

    client: Any



def maybe_create_langfuse(public_key: str | None, secret_key: str | None, host: str | None) -> LangfuseHandle | None:

    if not (public_key and secret_key and host):

        return None



    try:

        from langfuse import Langfuse  # type: ignore

    except Exception:

        return None



    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    return LangfuseHandle(client=client)



def start_trace(langfuse: LangfuseHandle | None, name: str, user_id: str | None, metadata: dict | None = None, tags: list[str] | None = None):

    if not langfuse:

        return None



    return langfuse.client.trace(

        name=name,

        user_id=user_id,

        metadata=metadata or {},

        tags=tags or [],

    )



def log_generation(trace, name: str, model: str, input_payload: dict, output_payload: dict, usage: dict | None = None, metadata: dict | None = None):

    if not trace:

        return None



    return trace.generation(

        name=name,

        model=model,

        input=input_payload,

        output=output_payload,

        usage=usage or {},

        metadata=metadata or {},

    )



def log_score(trace, name: str, value: float, comment: str | None = None, metadata: dict | None = None):

    if not trace:

        return None



    return trace.score(

        name=name,

        value=value,

        comment=comment,

        metadata=metadata or {},

    )



def flush(langfuse: LangfuseHandle | None):

    if not langfuse:

        return

    try:

        langfuse.client.flush()

    except Exception:

        pass
