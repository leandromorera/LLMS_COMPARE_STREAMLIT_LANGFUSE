"""Receipt Forgery Detection — Streamlit Dashboard."""
import os
import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# Try imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.voting import majority_vote, vote_tally
from src.image_features import image_basic_stats, blur_variance_of_laplacian, brightness_contrast
from src.ocr_parser import extract_total_from_text, load_ocr_text
from src.dataset import load_dataset, find_image_path, find_ocr_path, load_eval_cache, save_eval_cache
from src.judge import run_all_judges, JUDGE_CONFIGS

# Page config
st.set_page_config(page_title="Receipt Forgery Detector", page_icon="🧾", layout="wide")

# --- Sidebar ---
st.sidebar.title("🧾 Receipt Forgery Detector")
st.sidebar.markdown("---")

dataset_mode = st.sidebar.radio("Dataset mode", ["sample", "custom path"])

DATA_DIR = Path(__file__).parent / "data"

if dataset_mode == "sample":
    labels_path = str(DATA_DIR / "sample_labels.csv")
    image_dir = str(DATA_DIR / "sample" / "images")
    ocr_dir = str(DATA_DIR / "sample" / "ocr")
else:
    labels_path = st.sidebar.text_input("Labels CSV path", "")
    image_dir = st.sidebar.text_input("Image directory", "")
    ocr_dir = st.sidebar.text_input("OCR directory (optional)", "")

# Load dataset
@st.cache_data
def load_cached_dataset(lpath):
    if not lpath or not Path(lpath).exists():
        return []
    try:
        return load_dataset(lpath)
    except Exception as e:
        return []

dataset = load_cached_dataset(labels_path)
st.sidebar.write(f"Rows: {len(dataset)}")

if dataset:
    names = [r['image'] for r in dataset]
    selected_name = st.sidebar.selectbox("Receipt", names)
else:
    selected_name = None

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Receipt", "📊 Dataset Statistics", "⚡ Batch Evaluation"])

# ===================== TAB 1 =====================
with tab1:
    st.header("🔍 Analyze Receipt")

    if not selected_name:
        st.info("No dataset loaded. Please configure dataset in sidebar.")
    else:
        # Find image
        img_path = None
        if image_dir and Path(image_dir).exists():
            from src.dataset import find_image_path as _fip
            img_path = _fip(selected_name, image_dir)

        col1, col2 = st.columns(2)

        with col1:
            if img_path and Path(img_path).exists():
                st.image(img_path, caption=selected_name, use_container_width=True)
            else:
                st.warning(f"Image not found: {selected_name}")

        with col2:
            st.subheader("Image Features")
            if img_path and Path(img_path).exists():
                try:
                    stats = image_basic_stats(img_path)
                    blur = blur_variance_of_laplacian(img_path)
                    bc = brightness_contrast(img_path)
                    features = {**stats, "sharpness": round(blur, 2), **bc}
                    st.json(features)
                except Exception as e:
                    st.error(f"Feature extraction error: {e}")

            ocr_path = None
            ocr_text = None
            if ocr_dir:
                ocr_path = find_ocr_path(selected_name, ocr_dir)
                if ocr_path:
                    ocr_text = load_ocr_text(ocr_path)
                    total = extract_total_from_text(ocr_text) if ocr_text else None
                    if total:
                        st.metric("Extracted Total", f"${total:.2f}")

        st.markdown("---")

        if st.button("🔍 Run Judges", type="primary"):
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                st.error("OPENAI_API_KEY not set. Please add to .env file.")
            elif not img_path or not Path(img_path).exists():
                st.error("Image file not found.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")))

                    with st.spinner("Running 3 judges..."):
                        results = run_all_judges(client, img_path, ocr_text)

                    labels = [r["label"] for r in results]
                    verdict = majority_vote(labels)
                    tally = vote_tally(labels)

                    # Show verdict
                    color = {"FAKE": "🔴", "REAL": "🟢", "UNCERTAIN": "🟡"}.get(verdict, "⚪")
                    st.markdown(f"## {color} Verdict: **{verdict}**")
                    st.write("Vote tally:", tally)

                    # Per-judge results
                    cols = st.columns(3)
                    for i, (result, col) in enumerate(zip(results, cols)):
                        with col:
                            badge = {"FAKE": "🔴", "REAL": "🟢", "UNCERTAIN": "🟡"}.get(result["label"], "⚪")
                            st.markdown(f"### {badge} Judge {i+1}")
                            st.write(f"**Model:** {result.get('model', 'N/A')}")
                            st.write(f"**Label:** {result['label']}")
                            st.write(f"**Confidence:** {result['confidence']}%")
                            st.write(f"**Latency:** {result.get('latency_ms', 0)}ms")
                            with st.expander("Reasons"):
                                for r in result.get("reasons", []):
                                    st.write(f"• {r}")
                            if result.get("flags"):
                                with st.expander("Flags"):
                                    for f in result.get("flags", []):
                                        st.write(f"⚠️ {f}")

                    # Langfuse logging
                    try:
                        from src.langfuse_logger import log_receipt_trace
                        ground_truth = next((r['label'] for r in dataset if r['image'] == selected_name), None)
                        img_stats = image_basic_stats(img_path) if img_path else {}
                        log_receipt_trace(
                            image_path=img_path,
                            ground_truth=ground_truth,
                            image_stats=img_stats,
                            judge_results=results,
                            final_verdict=verdict,
                            vote_tally_result=tally,
                        )
                    except Exception:
                        pass

                except ImportError:
                    st.error("openai package not installed. Run: pip install openai")
                except Exception as e:
                    st.error(f"Error running judges: {e}")

# ===================== TAB 2 =====================
with tab2:
    st.header("📊 Dataset Statistics")

    if not dataset:
        st.info("No dataset loaded.")
    else:
        try:
            import pandas as pd

            df = pd.DataFrame(dataset)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Label Distribution")
                label_counts = df['label'].value_counts()
                st.bar_chart(label_counts)

            with col2:
                st.subheader("Summary")
                st.dataframe(df.groupby('label').size().reset_index(name='count'))

            # Image features scan
            if image_dir and Path(image_dir).exists():
                if st.button("📊 Scan Image Features"):
                    features_list = []
                    progress = st.progress(0)
                    for i, row in enumerate(dataset):
                        img_path = find_image_path(row['image'], image_dir)
                        if img_path:
                            try:
                                stats = image_basic_stats(img_path)
                                bc = brightness_contrast(img_path)
                                blur = blur_variance_of_laplacian(img_path)
                                features_list.append({
                                    'image': row['image'],
                                    'label': row['label'],
                                    **stats,
                                    **bc,
                                    'sharpness': round(blur, 2),
                                })
                            except Exception:
                                pass
                        progress.progress((i + 1) / len(dataset))

                    if features_list:
                        feat_df = pd.DataFrame(features_list)
                        st.subheader("Feature Statistics by Label")
                        st.dataframe(feat_df.groupby('label')[['width', 'height', 'file_size_kb', 'brightness', 'contrast', 'sharpness']].describe())

                        st.subheader("File Size Distribution")
                        st.bar_chart(feat_df.groupby('label')['file_size_kb'].mean())
        except ImportError:
            st.error("pandas required for statistics. Run: pip install pandas")

# ===================== TAB 3 =====================
with tab3:
    st.header("⚡ Batch Evaluation")

    if not dataset:
        st.info("No dataset loaded.")
    else:
        # Pool filter
        pool_filter = st.radio("Pool filter", ["All", "REAL only", "FAKE only"], horizontal=True)

        filtered = dataset.copy()
        if pool_filter == "REAL only":
            filtered = [r for r in filtered if r['label'] == 'REAL']
        elif pool_filter == "FAKE only":
            filtered = [r for r in filtered if r['label'] == 'FAKE']

        names = [r['image'] for r in filtered]

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎲 Random 5"):
                import random
                st.session_state['batch_selection'] = random.sample(names, min(5, len(names)))
        with col2:
            if st.button("✓ All"):
                st.session_state['batch_selection'] = names[:30]
        with col3:
            if st.button("✕ Clear"):
                st.session_state['batch_selection'] = []

        default_sel = st.session_state.get('batch_selection', [])
        selected_batch = st.multiselect("Select receipts", names, default=default_sel)

        run_name = st.text_input("Run name (for Langfuse)", f"batch-eval-{int(time.time())}")

        # Cost estimate
        if selected_batch:
            st.subheader("💰 Estimated Cost")
            n = len(selected_batch)
            cost_data = []
            for cfg in JUDGE_CONFIGS:
                model = os.environ.get(cfg["model_env"], cfg["default_model"])
                # Rough estimate: ~1000 tokens per image at $5/1M for gpt-4o
                est_cost = n * 0.005
                cost_data.append({"Judge": cfg["name"], "Model": model, "Receipts": n, "Est. Cost ($)": f"${est_cost:.3f}"})

            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(cost_data))
            except ImportError:
                for row in cost_data:
                    st.write(row)

        # Run batch
        if selected_batch and st.button("▶️ Run Batch Evaluation", type="primary"):
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                st.error("OPENAI_API_KEY not set.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")))

                    cache_path = str(DATA_DIR / "eval_cache.json")
                    cache = load_eval_cache(cache_path)

                    results_list = []
                    progress = st.progress(0)

                    for i, name in enumerate(selected_batch):
                        if name in cache:
                            results_list.append(cache[name])
                        else:
                            img_path = find_image_path(name, image_dir) if image_dir else None
                            if not img_path:
                                continue

                            ocr_path = find_ocr_path(name, ocr_dir) if ocr_dir else None
                            ocr_text = load_ocr_text(ocr_path) if ocr_path else None

                            judge_results = run_all_judges(client, img_path, ocr_text)
                            labels = [r["label"] for r in judge_results]
                            verdict = majority_vote(labels)
                            tally = vote_tally(labels)

                            ground_truth = next((r['label'] for r in dataset if r['image'] == name), None)

                            entry = {
                                "image": name,
                                "ground_truth": ground_truth,
                                "verdict": verdict,
                                "tally": tally,
                                "judges": judge_results,
                            }
                            cache[name] = entry
                            results_list.append(entry)
                            save_eval_cache(cache_path, cache)

                        progress.progress((i + 1) / len(selected_batch))

                    if results_list:
                        st.success(f"Evaluated {len(results_list)} receipts")

                        try:
                            import pandas as pd

                            res_df = pd.DataFrame([
                                {"image": r["image"], "ground_truth": r.get("ground_truth"), "verdict": r["verdict"]}
                                for r in results_list
                            ])

                            st.subheader("Results")
                            st.dataframe(res_df)

                            # Prediction distribution
                            st.subheader("Prediction Distribution")
                            st.bar_chart(res_df['verdict'].value_counts())

                            # Wrong predictions
                            wrong = res_df[res_df['ground_truth'].notna() & (res_df['ground_truth'] != res_df['verdict'])]
                            if not wrong.empty:
                                st.subheader("❌ Wrong Predictions")
                                st.dataframe(wrong)
                        except ImportError:
                            pass

                        # JSON download
                        json_str = json.dumps(results_list, indent=2, default=str)
                        st.download_button("📥 Download Results JSON", json_str, "eval_results.json", "application/json")

                except Exception as e:
                    st.error(f"Batch evaluation error: {e}")
