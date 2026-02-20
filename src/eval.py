from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)

class JudgeResult:

    judge_name: str

    model: str

    output: dict

    meta: dict



@dataclass(frozen=True)

class AggregatedResult:

    receipt_id: str

    final_label: str

    tally: dict

    judges: list[JudgeResult]

    ground_truth: str | None
