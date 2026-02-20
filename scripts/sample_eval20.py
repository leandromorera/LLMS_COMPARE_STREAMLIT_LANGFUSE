from __future__ import annotations



import argparse

from pathlib import Path



import pandas as pd



def main() -> None:

    p = argparse.ArgumentParser()

    p.add_argument("--labels", type=str, required=True, help="Path to train.txt (must include image, forged)")

    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--out_csv", type=str, default="eval_samples/eval_20.csv")

    args = p.parse_args()



    df = pd.read_csv(Path(args.labels))

    df["label"] = df["forged"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")



    fake = df[df["label"] == "FAKE"].sample(n=10, random_state=args.seed)

    real = df[df["label"] == "REAL"].sample(n=10, random_state=args.seed)

    out = pd.concat([fake, real], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)



    out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out[["image", "label"]].to_csv(out_path, index=False)



    print("Seed:", args.seed)

    print("Wrote:", out_path.resolve())

    print(out[["image", "label"]].to_string(index=False))



if __name__ == "__main__":

    main()
