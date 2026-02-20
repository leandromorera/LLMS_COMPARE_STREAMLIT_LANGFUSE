from __future__ import annotations



import argparse

import json

from pathlib import Path



import pandas as pd

from dotenv import load_dotenv



from openai import OpenAI



from src.config import load_settings

from src.judges import JudgeConfig, run_judge

from src.langfuse_logger import flush, log_generation, log_score, maybe_create_langfuse, start_trace

from src.vote import majority_vote, vote_tally



def main() -> None:

    load_dotenv()



    p = argparse.ArgumentParser()

    p.add_argument("--eval_csv", type=str, required=True, help="CSV with columns: image,label (10 REAL + 10 FAKE)")

    p.add_argument("--image_dir", type=str, required=True)

    p.add_argument("--out_json", type=str, default="reports/eval20_results.json")

    p.add_argument("--user_id", type=str, default="eval-user")

    args = p.parse_args()



    settings = load_settings()

    client = OpenAI(api_key=settings.openai_api_key)



    eval_df = pd.read_csv(Path(args.eval_csv))

    image_dir = Path(args.image_dir)



    langfuse = maybe_create_langfuse(settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host)



    judge_cfgs = [

        JudgeConfig(name="judge_1", model=settings.judge_models[0], temperature=0.2, persona="strict, skeptical, focuses on forensic inconsistencies"),

        JudgeConfig(name="judge_2", model=settings.judge_models[1], temperature=0.4, persona="balanced, looks for plausible printing/scan artifacts vs tampering"),

        JudgeConfig(name="judge_3", model=settings.judge_models[2], temperature=0.7, persona="lenient, assumes real unless clear signs of manipulation"),

    ]



    results = []

    correct = 0

    disagree_cases = []



    for _, row in eval_df.iterrows():

        image_id = row["image"]

        gt = str(row["label"]).upper().strip()

        image_path = image_dir / image_id



        trace = start_trace(

            langfuse,

            name="receipt_eval20",

            user_id=args.user_id,

            metadata={"receipt_id": image_id, "ground_truth": gt},

            tags=["receipt", "eval20"],

        )



        labels = []

        judge_outputs = []



        for cfg in judge_cfgs:

            parsed, meta = run_judge(client, cfg, image_path=image_path, receipt_id=image_id)

            labels.append(parsed["label"])

            judge_outputs.append({"judge": cfg.name, "model": cfg.model, **parsed})



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



        is_correct = 1.0 if final_label == gt else 0.0

        log_score(trace, name="final_correct", value=is_correct, comment=f"final={final_label} gt={gt}")



        if is_correct == 1.0:

            correct += 1



        if len(set(labels)) > 1:

            disagree_cases.append(

                {

                    "image": image_id,

                    "ground_truth": gt,

                    "final": final_label,

                    "tally": tally,

                    "judges": judge_outputs,

                }

            )



        results.append(

            {

                "image": image_id,

                "ground_truth": gt,

                "final": final_label,

                "tally": tally,

                "judges": judge_outputs,

            }

        )



    flush(langfuse)



    out_path = Path(args.out_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(

        {

            "accuracy": float(correct) / float(len(eval_df)) if len(eval_df) else 0.0,

            "n": int(len(eval_df)),

            "results": results,

            "disagreements_sample": disagree_cases[:3],

        },

        indent=2,

    ))



    print("Accuracy:", float(correct) / float(len(eval_df)) if len(eval_df) else 0.0)

    print("Wrote:", out_path.resolve())

    print("Disagreements captured:", len(disagree_cases))



if __name__ == "__main__":

    main()
