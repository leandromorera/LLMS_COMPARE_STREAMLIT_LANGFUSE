from __future__ import annotations

import base64

import json

from dataclasses import dataclass

from pathlib import Path

from typing import Any



from openai import OpenAI



_JSON_SCHEMA_HINT = {

    "type": "object",

    "additionalProperties": False,

    "properties": {

        "label": {"type": "string", "enum": ["FAKE", "REAL", "UNCERTAIN"]},

        "confidence": {"type": "number"},

        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},

        "flags": {"type": "array", "items": {"type": "string"}},

    },

    "required": ["label", "confidence", "reasons"],

}



@dataclass(frozen=True)

class JudgeConfig:

    name: str

    model: str

    temperature: float

    persona: str



def _image_to_data_url(image_path: Path) -> str:

    b = image_path.read_bytes()

    ext = image_path.suffix.lower().lstrip(".") or "png"

    mime = "png" if ext in ("png",) else "jpeg"

    b64 = base64.b64encode(b).decode("utf-8")

    return f"data:image/{mime};base64,{b64}"



def _base_prompt(persona: str) -> str:

    return (

        "You are an expert forensic document examiner evaluating whether a receipt image is forged.\n"

        f"Persona: {persona}\n\n"

        "Return ONLY valid JSON with this schema:\n"

        "{\n"

        '  "label": "FAKE|REAL|UNCERTAIN",\n'

        '  "confidence": 0.0,\n'

        '  "reasons": ["short reason 1", "short reason 2"],\n'

        '  "flags": ["optional tag", "optional tag"]\n'

        "}\n\n"

        "Guidelines:\n"

        "- Base your decision primarily on visual cues in the image (fonts, alignment, inconsistent spacing, copied/pasted digits, pixel artifacts, weird shadows, mismatched totals, etc.).\n"

        "- If evidence is weak or image quality is poor, use UNCERTAIN and lower confidence.\n"

        "- confidence is 0 to 100.\n"

        "- reasons must be short, concrete, and observable.\n"

    )



def run_judge(client: OpenAI, cfg: JudgeConfig, image_path: Path, receipt_id: str) -> tuple[dict, dict]:

    prompt = _base_prompt(cfg.persona)

    data_url = _image_to_data_url(image_path)



    messages: list[dict[str, Any]] = [

        {

            "role": "system",

            "content": "You are a careful evaluator that outputs strict JSON.",

        },

        {

            "role": "user",

            "content": [

                {"type": "text", "text": f"Receipt ID: {receipt_id}\n\n{prompt}"},

                {"type": "image_url", "image_url": {"url": data_url}},

            ],

        },

    ]



    # Prefer structured JSON outputs when supported by the model.

    try:

        resp = client.chat.completions.create(

            model=cfg.model,

            temperature=cfg.temperature,

            messages=messages,

            response_format={"type": "json_object"},

        )

        raw = resp.choices[0].message.content or "{}"

        usage = {

            "input_tokens": getattr(resp.usage, "prompt_tokens", None),

            "output_tokens": getattr(resp.usage, "completion_tokens", None),

            "total_tokens": getattr(resp.usage, "total_tokens", None),

        }

    except Exception:

        resp = client.chat.completions.create(

            model=cfg.model,

            temperature=cfg.temperature,

            messages=messages,

        )

        raw = resp.choices[0].message.content or "{}"

        usage = {

            "input_tokens": getattr(resp.usage, "prompt_tokens", None),

            "output_tokens": getattr(resp.usage, "completion_tokens", None),

            "total_tokens": getattr(resp.usage, "total_tokens", None),

        }



    parsed = _safe_parse_json(raw)

    parsed = _normalize_output(parsed)



    input_payload = {

        "receipt_id": receipt_id,

        "prompt": prompt,

        "model": cfg.model,

        "temperature": cfg.temperature,

        "persona": cfg.persona,

        "image_path": str(image_path),

    }

    output_payload = {"raw": raw, "parsed": parsed}



    return parsed, {"usage": usage, "input": input_payload, "output": output_payload}



def _safe_parse_json(text: str) -> dict:

    try:

        return json.loads(text)

    except Exception:

        # naive salvage: grab the first {...} block

        start = text.find("{")

        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:

            try:

                return json.loads(text[start : end + 1])

            except Exception:

                return {}

        return {}



def _normalize_output(d: dict) -> dict:

    label = str(d.get("label", "UNCERTAIN")).upper().strip()

    if label not in ("FAKE", "REAL", "UNCERTAIN"):

        label = "UNCERTAIN"



    try:

        conf = float(d.get("confidence", 0.0))

    except Exception:

        conf = 0.0

    conf = max(0.0, min(100.0, conf))



    reasons = d.get("reasons") or []

    if isinstance(reasons, str):

        reasons = [reasons]

    if not isinstance(reasons, list):

        reasons = []

    reasons = [str(x)[:240] for x in reasons if str(x).strip()][:5] or ["Insufficient evidence in image."]



    flags = d.get("flags") or []

    if isinstance(flags, str):

        flags = [flags]

    if not isinstance(flags, list):

        flags = []

    flags = [str(x)[:64] for x in flags if str(x).strip()][:8]



    return {

        "label": label,

        "confidence": conf,

        "reasons": reasons,

        "flags": flags,

    }
