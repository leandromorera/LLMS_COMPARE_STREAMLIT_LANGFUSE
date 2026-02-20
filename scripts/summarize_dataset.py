from __future__ import annotations



import argparse

from pathlib import Path



import matplotlib.pyplot as plt

import pandas as pd



from src.dataset import extract_total_from_text, image_basic_stats

from src.features import blur_variance_of_laplacian, brightness_contrast



def main() -> None:

    p = argparse.ArgumentParser()

    p.add_argument("--labels", type=str, required=True, help="Path to train.txt or labels.csv (must include image, forged)")

    p.add_argument("--image_dir", type=str, required=True, help="Directory containing receipt images")

    p.add_argument("--ocr_dir", type=str, default="", help="Directory containing per-image OCR .txt (optional)")

    p.add_argument("--out_dir", type=str, default="reports")

    args = p.parse_args()



    labels_path = Path(args.labels)

    image_dir = Path(args.image_dir)

    ocr_dir = Path(args.ocr_dir) if args.ocr_dir else None

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    df = pd.read_csv(labels_path)

    df["label"] = df["forged"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")



    # Basic counts

    counts = df["label"].value_counts().to_dict()

    (out_dir / "counts.json").write_text(pd.Series(counts).to_json())



    # Image stats + optional totals

    rows = []

    for _, r in df.iterrows():

        image_id = r["image"]

        label = r["label"]

        img_path = image_dir / image_id

        if not img_path.exists():

            continue



        stats = image_basic_stats(img_path)

        stats.update(brightness_contrast(img_path))

        stats["blur_var_laplacian"] = blur_variance_of_laplacian(img_path)

        stats["image"] = image_id

        stats["label"] = label



        total = None

        if ocr_dir is not None:

            txt_path = (ocr_dir / Path(image_id).with_suffix(".txt").name)

            if txt_path.exists():

                total = extract_total_from_text(txt_path.read_text(errors="ignore"))

        stats["total_amount"] = total

        rows.append(stats)



    feat = pd.DataFrame(rows)

    feat.to_csv(out_dir / "dataset_features.csv", index=False)



    # Plot: totals histogram (only if available)

    if feat["total_amount"].notna().any():

        x = feat["total_amount"].dropna()

        plt.figure()

        plt.hist(x, bins=20)

        plt.title("Receipt totals distribution")

        plt.xlabel("Total amount (parsed from OCR/text)")

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(out_dir / "totals_hist.png", dpi=150)

        plt.close()



    # Plot: image resolution distribution

    plt.figure()

    plt.hist(feat["width"], bins=20)

    plt.title("Image width distribution")

    plt.xlabel("Width (px)")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(out_dir / "width_hist.png", dpi=150)

    plt.close()



    plt.figure()

    plt.hist(feat["height"], bins=20)

    plt.title("Image height distribution")

    plt.xlabel("Height (px)")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(out_dir / "height_hist.png", dpi=150)

    plt.close()



    # Plot: label counts

    plt.figure()

    plt.bar(list(counts.keys()), list(counts.values()))

    plt.title("REAL vs FAKE count")

    plt.xlabel("Label")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(out_dir / "label_counts.png", dpi=150)

    plt.close()



    print("Wrote reports to:", out_dir.resolve())



if __name__ == "__main__":

    main()
