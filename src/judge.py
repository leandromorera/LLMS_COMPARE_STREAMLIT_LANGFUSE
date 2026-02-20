"""LLM Judge Panel: 3 independent OpenAI vision judges."""
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

JUDGE_CONFIGS = [
    {
        "name": "judge_1",
        "model_env": "JUDGE_MODEL_1",
        "default_model": "gpt-4o",
        "persona": "strict/skeptical",
        "temperature": 0.2,
    },
    {
        "name": "judge_2",
        "model_env": "JUDGE_MODEL_2",
        "default_model": "gpt-4o",
        "persona": "balanced/artifact-aware",
        "temperature": 0.4,
    },
    {
        "name": "judge_3",
        "model_env": "JUDGE_MODEL_3",
        "default_model": "gpt-4o",
        "persona": "lenient/benefit-of-doubt",
        "temperature": 0.7,
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are a forensic document analyst acting as a {persona} receipt authenticity judge.
Analyze the provided receipt image and classify it as FAKE, REAL, or UNCERTAIN.

You must respond in valid JSON with this exact schema:
{{
  "label": "FAKE" | "REAL" | "UNCERTAIN",
  "confidence": <integer 0-100>,
  "reasons": [<string>, ...],
  "flags": [<string>, ...]
}}

Rules:
- label must be exactly one of: FAKE, REAL, UNCERTAIN
- confidence is 0-100 integer
- reasons: list of observations supporting your classification
- flags: list of specific forgery indicators (empty list if REAL)
- Use UNCERTAIN when evidence is genuinely ambiguous
"""


def _encode_image(image_path: str) -> str:
    """Base64 encode image for OpenAI vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(image_path: str) -> str:
    """Determine media type from file extension."""
    ext = Path(image_path).suffix.lower()
    return "image/png" if ext == ".png" else "image/jpeg"


def _normalize_output(raw: dict) -> dict:
    """Enforce schema constraints on judge output."""
    valid_labels = {"FAKE", "REAL", "UNCERTAIN"}
    label = str(raw.get("label", "UNCERTAIN")).upper()
    if label not in valid_labels:
        label = "UNCERTAIN"

    confidence = raw.get("confidence", 50)
    try:
        confidence = int(confidence)
        confidence = max(0, min(100, confidence))
    except (TypeError, ValueError):
        confidence = 50

    reasons = raw.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons]

    flags = raw.get("flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(f) for f in flags]

    return {"label": label, "confidence": confidence, "reasons": reasons, "flags": flags}


def _extract_json_block(text: str) -> dict:
    """Extract first JSON object from text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def run_judge(
    client,
    image_path: str,
    judge_config: dict,
    ocr_text: Optional[str] = None,
) -> dict:
    """Run a single judge on an image. Returns normalized result + latency_ms."""
    model = os.environ.get(judge_config["model_env"], judge_config["default_model"])
    persona = judge_config["persona"]
    temperature = judge_config["temperature"]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(persona=persona)

    image_b64 = _encode_image(image_path)
    media_type = _image_media_type(image_path)

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        }
    ]

    if ocr_text:
        user_content.append({
            "type": "text",
            "text": f"OCR text from receipt:\n{ocr_text[:2000]}"
        })

    user_content.append({
        "type": "text",
        "text": "Classify this receipt as FAKE, REAL, or UNCERTAIN. Return JSON only."
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        raw_text = response.choices[0].message.content
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            raw = _extract_json_block(raw_text)

        usage = response.usage
        token_counts = {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
    except Exception as e:
        raw = {"label": "UNCERTAIN", "confidence": 0, "reasons": [str(e)], "flags": []}
        token_counts = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    latency_ms = int((time.time() - start) * 1000)
    result = _normalize_output(raw)
    result["latency_ms"] = latency_ms
    result["model"] = model
    result["judge_name"] = judge_config["name"]
    result.update(token_counts)
    return result


def run_all_judges(
    client,
    image_path: str,
    ocr_text: Optional[str] = None,
) -> list[dict]:
    """Run all 3 judges and return list of results."""
    results = []
    for config in JUDGE_CONFIGS:
        result = run_judge(client, image_path, config, ocr_text)
        results.append(result)
    return results
