from __future__ import annotations



import argparse

import json

from pathlib import Path



from dotenv import load_dotenv



from openai import OpenAI



from src.config import load_settings

from src.dataset import extract_total_from_text

from src.judges import JudgeConfig, run_judge

from src.langfuse_logger import flush, log_generation, log_score, maybe_create_langfuse, start_trace

from src.vote import majority_vote, vote_tally



def main() -> None:

    load_dotenv()



    p = argparse.ArgumentParser()

    p.add_argument("--image", type=str, required=True)

    p.add_argument("--ocr", type=str, default="")

    p.add_argument("--receipt_id", type=str, default="")

    p.add_argument("--ground_truth", type=str, default="")  # optional, for evaluation only

    p.add_argument("--user_id", type=str, default="demo-user")

    args = p.parse_args()



    settings = load_settings()

    client = OpenAI(api_key=settings.openai_api_key)



    image_path = Path(args.image)

    receipt_id = args.receipt_id or image_path.name

    ground_truth = args.ground_truth.strip().upper() or None



    ocr_text = None

    if args.ocr:

        ocr_path = Path(args.ocr)

        if ocr_path.exists():

            ocr_text = ocr_path.read_text(errors="ignore")



    extracted_total = extract_total_from_text(ocr_text) if ocr_text else None



    langfuse = maybe_create_langfuse(settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host)

    trace = start_trace(

        langfuse,

        name="receipt_judging",

        user_id=args.user_id,

        metadata={

            "receipt_id": receipt_id,

            "ground_truth": ground_truth,

            "extracted_total": extracted_total,

        },

        tags=["receipt", "llm-judge"],

    )



    judge_cfgs = [

        JudgeConfig(name="judge_1", model=settings.judge_models[0], temperature=0.2, persona="strict, skeptical, focuses on forensic inconsistencies"),

        JudgeConfig(name="judge_2", model=settings.judge_models[1], temperature=0.4, persona="balanced, looks for plausible printing/scan artifacts vs tampering"),

        JudgeConfig(name="judge_3", model=settings.judge_models[2], temperature=0.7, persona="lenient, assumes real unless clear signs of manipulation"),

    ]



    outputs = []

    labels = []



    for cfg in judge_cfgs:

        parsed, meta = run_judge(client, cfg, image_path=image_path, receipt_id=receipt_id)

        outputs.append({"judge": cfg.name, "model": cfg.model, **parsed})

        labels.append(parsed["label"])



        log_generation(

            trace,

            name=cfg.name,

            model=cfg.model,

            input_payload=meta["input"],

            output_payload=meta["output"],

            usage=meta["usage"],

            metadata={"temperature": cfg.temperature},

        )



        log_score(trace, name=f"{cfg.name}_confidence", value=float(parsed["confidence"]), comment=parsed["label"])



    final_label = majority_vote(labels)

    tally = vote_tally(labels)



    # Evaluation score (if ground truth provided)

    if ground_truth in ("FAKE", "REAL"):

        correct = 1.0 if final_label == ground_truth else 0.0

        log_score(trace, name="final_correct", value=correct, comment=f"final={final_label} gt={ground_truth}")



    flush(langfuse)



    result = {

        "receipt_id": receipt_id,

        "final_label": final_label,

        "tally": tally,

        "judges": outputs,

        "ground_truth": ground_truth,

        "extracted_total": extracted_total,

    }



    print(json.dumps(result, indent=2))



if __name__ == "__main__":

    main()
