from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.config import load_settings
from src.dataset import extract_total_from_text, image_basic_stats
from src.features import blur_variance_of_laplacian, brightness_contrast
from src.judges import JudgeConfig, run_judge
from src.langfuse_logger import flush, log_generation, log_score, maybe_create_langfuse, start_trace
from src.langfuse_mcp_client import LangfuseMCPClient
from src.vote import majority_vote, vote_tally

st.set_page_config(page_title="LLM-Judge Fake Receipt Detector", layout="wide")


# ---------------------------------------------------------------------------
# Dataset-wide stats (cached so scanning 500+ images only happens once)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _compute_dataset_stats(label_path_str: str, image_dir_str: str, ocr_dir_str: str) -> pd.DataFrame:
    """Return a DataFrame with per-image stats for every row in the label file."""
    label_path = Path(label_path_str)
    image_dir = Path(image_dir_str)
    ocr_dir = Path(ocr_dir_str) if ocr_dir_str else None

    df = pd.read_csv(label_path)
    if "label" not in df.columns and "forged" in df.columns:
        df["label"] = df["forged"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")

    rows = []
    for _, row in df.iterrows():
        img_id = str(row["image"])
        label = str(row.get("label", ""))
        img_path = image_dir / img_id
        entry: dict = {"image": img_id, "label": label}

        if img_path.exists():
            entry.update(image_basic_stats(img_path))
            entry.update(brightness_contrast(img_path))
            entry["blur_var_laplacian"] = blur_variance_of_laplacian(img_path)

        if ocr_dir is not None:
            ocr_path = ocr_dir / Path(img_id).with_suffix(".txt").name
            if ocr_path.exists():
                text = ocr_path.read_text(errors="ignore")
                entry["total_amount"] = extract_total_from_text(text)

        rows.append(entry)

    return pd.DataFrame(rows)


_LABEL_COLORS = {"REAL": "#2ecc71", "FAKE": "#e74c3c"}
_PLOTLY_THEME = "plotly_white"

# Pricing in USD per 1 million tokens (approximate, Feb 2026)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
    "gpt-4.1":      {"input": 2.00,  "output": 8.00},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
}
_TOKENS_PER_IMAGE_INPUT  = 1_500  # system prompt + image + user message (estimate)
_TOKENS_PER_IMAGE_OUTPUT = 180    # JSON response (estimate)


def _estimate_cost(n_images: int, models: list[str]) -> list[dict]:
    """Return per-judge cost estimate rows for n_images receipts."""
    rows = []
    for i, model in enumerate(models, 1):
        pricing = _MODEL_PRICING.get(model, {"input": 2.50, "output": 10.00})
        inp_tokens = n_images * _TOKENS_PER_IMAGE_INPUT
        out_tokens = n_images * _TOKENS_PER_IMAGE_OUTPUT
        cost = (inp_tokens * pricing["input"] + out_tokens * pricing["output"]) / 1_000_000
        rows.append({
            "Judge":               f"judge_{i}",
            "Model":               model,
            "Est. input tokens":   inp_tokens,
            "Est. output tokens":  out_tokens,
            "Est. cost (USD)":     round(cost, 5),
        })
    return rows


def _hist(df: pd.DataFrame, col: str, title: str, xlabel: str,
          by_label: bool = True) -> go.Figure:
    """Return a Plotly histogram, optionally split by REAL/FAKE label."""
    if by_label:
        fig = px.histogram(
            df.dropna(subset=[col]),
            x=col, color="label",
            color_discrete_map=_LABEL_COLORS,
            barmode="overlay", opacity=0.72,
            nbins=30, title=title,
            labels={col: xlabel},
            template=_PLOTLY_THEME,
        )
    else:
        fig = px.histogram(
            df.dropna(subset=[col]),
            x=col, nbins=30, title=title,
            labels={col: xlabel},
            template=_PLOTLY_THEME,
            color_discrete_sequence=["#3498db"],
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


def _stats_tab(label_path: Path, image_dir: Path, ocr_dir: Path | None) -> None:
    st.subheader("Dataset Statistics")

    ocr_str = str(ocr_dir) if ocr_dir else ""
    with st.spinner("Scanning dataset… (cached after first run)"):
        stats = _compute_dataset_stats(str(label_path), str(image_dir), ocr_str)

    # ── 1. REAL vs FAKE ──────────────────────────────────────────────────────
    st.markdown("#### Label distribution")
    counts = stats["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]

    m1, m2, m3 = st.columns([1, 1, 3])
    m1.metric("REAL", int(stats["label"].eq("REAL").sum()))
    m2.metric("FAKE", int(stats["label"].eq("FAKE").sum()))
    with m3:
        fig = px.bar(
            counts, x="label", y="count", color="label",
            color_discrete_map=_LABEL_COLORS,
            text="count", template=_PLOTLY_THEME,
            title="REAL vs FAKE receipt count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 2. Receipt totals ─────────────────────────────────────────────────────
    if "total_amount" in stats.columns and stats["total_amount"].notna().any():
        st.markdown("#### Receipt totals")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _hist(stats, "total_amount", "Total amount — all receipts",
                      "Amount", by_label=False),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _hist(stats, "total_amount", "Total amount by label — are fakes skewed?",
                      "Amount", by_label=True),
                use_container_width=True,
            )
    else:
        st.info("No OCR directory provided — receipt total distribution not available.")

    st.divider()

    # ── 3. Image dimensions & file size ──────────────────────────────────────
    if "width" in stats.columns:
        st.markdown("#### Image dimensions & file size")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.histogram(
                stats, x="width", y="height", color="label",
                color_discrete_map=_LABEL_COLORS,
                marginal="rug", opacity=0.7,
                title="Width vs Height scatter (size proxy)",
                template=_PLOTLY_THEME,
                labels={"width": "Width (px)", "height": "Height (px)"},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(
                _hist(stats, "file_kb", "File size distribution (KB)", "KB", by_label=False),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(
                _hist(stats, "file_kb", "File size by label", "KB", by_label=True),
                use_container_width=True,
            )

    # ── 4. Aspect ratio ───────────────────────────────────────────────────────
    if "aspect_ratio" in stats.columns:
        st.markdown("#### Aspect ratio  (width ÷ height)")
        fig = _hist(stats, "aspect_ratio",
                    "Aspect ratio by label  ·  >1 = landscape  ·  <1 = portrait",
                    "Aspect ratio", by_label=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 5. Image quality proxies ──────────────────────────────────────────────
    if "blur_var_laplacian" in stats.columns:
        st.markdown("#### Image quality proxies  (REAL vs FAKE)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                _hist(stats, "blur_var_laplacian",
                      "Sharpness — Var. of Laplacian", "Higher = sharper"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _hist(stats, "brightness_mean", "Brightness (mean pixel)", "0–1 scale"),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(
                _hist(stats, "contrast_std", "Contrast (pixel std-dev)", "0–1 scale"),
                use_container_width=True,
            )

    st.divider()

    # ── 6. Summary table ──────────────────────────────────────────────────────
    st.markdown("#### Summary statistics by label")
    num_cols = [c for c in [
        "total_amount", "file_kb", "width", "height", "aspect_ratio",
        "blur_var_laplacian", "brightness_mean", "contrast_std",
    ] if c in stats.columns]
    if num_cols:
        st.dataframe(
            stats.groupby("label")[num_cols].describe().round(2),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Native file / folder picker (tkinter — local app only)
# ---------------------------------------------------------------------------

def _pick_file(title: str, filetypes: list) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path or ""


def _pick_dir(title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path or ""


def _browse_field(
    sidebar_label: str,
    state_key: str,
    default: str,
    btn_key: str,
    is_dir: bool = False,
    filetypes: list | None = None,
) -> str:
    """Render a sidebar text-input + Browse button; return the current value.

    Uses a separate backing key so the browse button can freely update
    session state without conflicting with the widget's own key.
    """
    backing = f"{state_key}__val"
    if backing not in st.session_state:
        st.session_state[backing] = default

    st.sidebar.markdown(f"**{sidebar_label}**")
    col_in, col_btn = st.sidebar.columns([5, 1])
    with col_in:
        # No key= on the widget — we own the backing var ourselves
        typed = st.text_input(
            sidebar_label,
            value=st.session_state[backing],
            label_visibility="collapsed",
        )
        st.session_state[backing] = typed  # keep in sync when user edits
    with col_btn:
        if st.button("📂", key=btn_key, help=f"Browse for {sidebar_label}"):
            path = _pick_dir(f"Select {sidebar_label}") if is_dir else _pick_file(
                f"Select {sidebar_label}", filetypes or [("All files", "*.*")]
            )
            if path:
                st.session_state[backing] = path
                st.rerun()
    return str(st.session_state[backing])


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

def _batch_results_charts(results: list[dict]) -> None:
    """Render all statistics charts for a completed batch evaluation."""
    if not results:
        st.info("No results to display.")
        return

    # ── Build flat DataFrames ─────────────────────────────────────────────────
    rows_main: list[dict] = []
    rows_judge: list[dict] = []
    for r in results:
        rows_main.append({
            "image":          r["image_id"],
            "ground_truth":   r["ground_truth"],
            "final_label":    r["final_label"],
            "is_correct":     r.get("is_correct", False),
            "tally_fake":     r["tally"].get("FAKE", 0),
            "tally_real":     r["tally"].get("REAL", 0),
            "tally_uncertain":r["tally"].get("UNCERTAIN", 0),
        })
        for j in r.get("judges", []):
            rows_judge.append({
                "image":        r["image_id"],
                "ground_truth": r["ground_truth"],
                "judge":        j["name"],
                "model":        j["model"],
                "label":        j["label"],
                "confidence":   j["confidence"],
                "is_correct":   j["label"] == r["ground_truth"],
            })

    df_main  = pd.DataFrame(rows_main)
    df_judge = pd.DataFrame(rows_judge)

    n          = len(df_main)
    n_correct  = int(df_main["is_correct"].sum())
    accuracy   = n_correct / n if n else 0.0
    avg_conf   = float(df_judge["confidence"].mean()) if not df_judge.empty else 0.0
    n_disagree = int(
        df_main.apply(
            lambda row: row["tally_fake"] > 0 and row["tally_real"] > 0, axis=1
        ).sum()
    )
    agree_rate = 1.0 - n_disagree / n if n else 1.0

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("#### Summary metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images evaluated", n)
    c2.metric("Accuracy", f"{accuracy:.1%}")
    c3.metric("Correct", n_correct)
    c4.metric("Avg. confidence", f"{avg_conf:.1f}")
    c5.metric("Agreement rate", f"{agree_rate:.1%}")

    st.divider()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("#### Confusion matrix  (final majority vote)")
    pred_labels = ["FAKE", "REAL", "UNCERTAIN"]
    gt_labels   = sorted(df_main["ground_truth"].unique())
    cm_data = [
        [int(((df_main["ground_truth"] == gt) & (df_main["final_label"] == pred)).sum())
         for pred in pred_labels]
        for gt in gt_labels
    ]
    fig_cm = px.imshow(
        cm_data,
        x=pred_labels, y=gt_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Ground truth (y) vs Predicted (x)",
        labels={"x": "Predicted", "y": "Ground truth", "color": "Count"},
        template=_PLOTLY_THEME,
    )
    fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))

    # ── Prediction distribution ───────────────────────────────────────────────
    pred_counts = df_main["final_label"].value_counts().reset_index()
    pred_counts.columns = ["label", "count"]
    color_map = {**_LABEL_COLORS, "UNCERTAIN": "#f39c12"}
    fig_pred = px.bar(
        pred_counts, x="label", y="count", color="label",
        color_discrete_map=color_map,
        text="count", title="Final prediction distribution",
        template=_PLOTLY_THEME,
    )
    fig_pred.update_traces(textposition="outside")
    fig_pred.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=40, b=10))

    col_cm, col_pred = st.columns(2)
    with col_cm:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_pred:
        st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()

    # ── Confidence box plots & histograms ─────────────────────────────────────
    if not df_judge.empty:
        st.markdown("#### Confidence by judge & ground truth")
        fig_box = px.box(
            df_judge,
            x="judge", y="confidence", color="ground_truth",
            color_discrete_map=_LABEL_COLORS,
            title="Confidence distribution per judge  (REAL vs FAKE)",
            template=_PLOTLY_THEME,
            points="all",
        )
        fig_box.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("#### Confidence histograms per judge")
        judge_names = sorted(df_judge["judge"].unique())
        hist_cols = st.columns(len(judge_names))
        for col, jname in zip(hist_cols, judge_names):
            with col:
                fig_h = px.histogram(
                    df_judge[df_judge["judge"] == jname],
                    x="confidence", color="ground_truth",
                    color_discrete_map=_LABEL_COLORS,
                    barmode="overlay", opacity=0.72,
                    nbins=20, title=jname,
                    template=_PLOTLY_THEME,
                )
                fig_h.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10),
                                    showlegend=False)
                st.plotly_chart(fig_h, use_container_width=True)

        st.divider()

        # ── Per-judge accuracy table ──────────────────────────────────────────
        st.markdown("#### Per-judge performance")
        judge_stats = (
            df_judge.groupby("judge")
            .agg(
                model=("model", "first"),
                accuracy=("is_correct", "mean"),
                avg_confidence=("confidence", "mean"),
                n_fake=("label", lambda s: (s == "FAKE").sum()),
                n_real=("label", lambda s: (s == "REAL").sum()),
                n_uncertain=("label", lambda s: (s == "UNCERTAIN").sum()),
            )
            .reset_index()
        )
        judge_stats["accuracy"]       = judge_stats["accuracy"].apply(lambda x: f"{x:.1%}")
        judge_stats["avg_confidence"] = judge_stats["avg_confidence"].round(1)
        st.dataframe(judge_stats, use_container_width=True, hide_index=True)

    st.divider()

    # ── Wrong predictions ─────────────────────────────────────────────────────
    wrong = df_main[~df_main["is_correct"]].copy()
    st.markdown(f"#### Wrong predictions  ({len(wrong)} / {n})")
    if wrong.empty:
        st.success("Perfect score — all predictions matched ground truth!")
    else:
        wrong_display = wrong[
            ["image", "ground_truth", "final_label",
             "tally_fake", "tally_real", "tally_uncertain"]
        ].copy()
        wrong_display.columns = [
            "Image", "Ground truth", "Predicted",
            "FAKE votes", "REAL votes", "UNCERTAIN votes",
        ]
        st.dataframe(wrong_display, use_container_width=True, hide_index=True)


def _batch_tab(
    labels_df: pd.DataFrame,
    image_dir: Path,
    ocr_dir: Path | None,
    client: OpenAI,
    judge_cfgs: list[JudgeConfig],
    settings,
) -> None:
    st.subheader("Batch Evaluation")
    st.caption("Run all 3 judges on a subset of the dataset and analyse the aggregate results.")

    n_total      = len(labels_df)
    n_real_total = int((labels_df["label"] == "REAL").sum())
    n_fake_total = int((labels_df["label"] == "FAKE").sum())

    # ── Pool filter + quick-select helpers ────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([2, 4])

    with ctrl_left:
        label_filter = st.radio(
            "Filter pool by label",
            ["All", "REAL only", "FAKE only"],
            horizontal=True,
        )
        log_langfuse = st.checkbox(
            "Log traces to Langfuse (requires MCP server)", value=False
        )

    if label_filter == "REAL only":
        pool_df = labels_df[labels_df["label"] == "REAL"].reset_index(drop=True)
    elif label_filter == "FAKE only":
        pool_df = labels_df[labels_df["label"] == "FAKE"].reset_index(drop=True)
    else:
        pool_df = labels_df.reset_index(drop=True)

    # Options shown in the multiselect: "filename (LABEL)"
    all_options = [f"{r['image']} ({r['label']})" for _, r in pool_df.iterrows()]

    with ctrl_right:
        st.markdown("**Quick select**")
        qs_col1, qs_col2, qs_col3, qs_col4 = st.columns([3, 1, 1, 1])
        n_quick = qs_col1.slider(
            "N images",
            min_value=1,
            max_value=max(len(pool_df), 1),
            value=min(20, len(pool_df)),
            label_visibility="collapsed",
        )
        if qs_col2.button("🎲 Random", help="Pick N random images from the filtered pool"):
            sampled = pool_df.sample(min(n_quick, len(pool_df)))
            st.session_state["batch_multiselect"] = [
                f"{r['image']} ({r['label']})" for _, r in sampled.iterrows()
            ]
            st.rerun()
        if qs_col3.button("✓ All", help="Select every image in the filtered pool"):
            st.session_state["batch_multiselect"] = all_options
            st.rerun()
        if qs_col4.button("✕ Clear", help="Deselect all"):
            st.session_state["batch_multiselect"] = []
            st.rerun()

    # ── Langfuse run name (shown only when logging is enabled) ────────────────
    run_name = "batch-eval"
    if log_langfuse:
        run_name = st.text_input(
            "Langfuse run name",
            value="batch-eval",
            key="batch_lf_run_name",
            help=(
                "Identifies this batch in Langfuse → Datasets → Runs. "
                "Use a different name for each experiment to compare them side-by-side."
            ),
        )

    # ── Exact multiselect ─────────────────────────────────────────────────────
    if "batch_multiselect" not in st.session_state:
        st.session_state["batch_multiselect"] = all_options[:min(20, len(all_options))]

    selected_options: list[str] = st.multiselect(
        "Selected images — type to search, click × to remove, click an option to add",
        options=all_options,
        key="batch_multiselect",
        placeholder="Start typing a filename…",
    )

    # Parse back to a DataFrame (preserves original row order)
    selected_ids = {opt.rsplit(" (", 1)[0] for opt in selected_options}
    selected_df  = labels_df[labels_df["image"].isin(selected_ids)].reset_index(drop=True)

    # ── Thumbnail grid of selected images ─────────────────────────────────────
    n_sel = len(selected_df)
    if n_sel == 0:
        st.info("No images selected. Use the filter + quick-select above, or type in the box.")
    else:
        n_real_sel = int((selected_df["label"] == "REAL").sum())
        n_fake_sel = int((selected_df["label"] == "FAKE").sum())
        st.markdown(
            f"**{n_sel} images selected** — "
            f"<span style='color:{_LABEL_COLORS['REAL']};font-weight:700'>REAL: {n_real_sel}</span>"
            f" · "
            f"<span style='color:{_LABEL_COLORS['FAKE']};font-weight:700'>FAKE: {n_fake_sel}</span>",
            unsafe_allow_html=True,
        )
        MAX_THUMBS   = 30
        COLS_PER_ROW = 6
        thumb_df = selected_df.head(MAX_THUMBS)
        if n_sel > MAX_THUMBS:
            st.caption(f"Showing first {MAX_THUMBS} of {n_sel} thumbnails.")
        for row_start in range(0, len(thumb_df), COLS_PER_ROW):
            thumb_cols = st.columns(COLS_PER_ROW)
            chunk = thumb_df.iloc[row_start : row_start + COLS_PER_ROW]
            for col_widget, (_, img_row) in zip(thumb_cols, chunk.iterrows()):
                img_path = image_dir / img_row["image"]
                color    = _LABEL_COLORS.get(img_row["label"], "#888")
                with col_widget:
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.markdown("_not found_")
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.68rem;"
                        f"color:{color};font-weight:700;margin-top:-4px'>"
                        f"{img_row['label']}</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Cost estimation ───────────────────────────────────────────────────────
    st.markdown("#### Estimated API cost")
    models    = [cfg.model for cfg in judge_cfgs]
    cost_rows = _estimate_cost(len(selected_df), models)
    cost_df   = pd.DataFrame(cost_rows)
    total_cost = cost_df["Est. cost (USD)"].sum()
    totals_row = pd.DataFrame([{
        "Judge":              "TOTAL",
        "Model":              f"{len(judge_cfgs)} judges  {len(selected_df)} images",
        "Est. input tokens":  int(cost_df["Est. input tokens"].sum()),
        "Est. output tokens": int(cost_df["Est. output tokens"].sum()),
        "Est. cost (USD)":    round(total_cost, 5),
    }])
    st.dataframe(pd.concat([cost_df, totals_row], ignore_index=True),
                 use_container_width=True, hide_index=True)
    st.caption(
        f"Estimates assume ~{_TOKENS_PER_IMAGE_INPUT:,} input tokens and "
        f"~{_TOKENS_PER_IMAGE_OUTPUT} output tokens per image per judge. "
        "Actual cost depends on image size and response length."
    )

    st.divider()

    # ── Run / clear buttons ───────────────────────────────────────────────────
    run_col, clear_col = st.columns([4, 1])
    run_clicked = run_col.button(
        f"  Run Batch Evaluation  ({len(selected_df)} images  {len(judge_cfgs)} judges)",
        type="primary",
        disabled=len(selected_df) == 0,
    )
    if clear_col.button("🗑 Clear", disabled="batch_results" not in st.session_state):
        del st.session_state["batch_results"]
        st.rerun()

    # ── Evaluation loop ───────────────────────────────────────────────────────
    if run_clicked:
        lf = None
        if log_langfuse:
            try:
                lf = LangfuseMCPClient()
                lf.auth_check()
            except Exception as exc:
                st.warning(f"Langfuse MCP not reachable ({exc}). Running without logging.")
                lf = None

        prog       = st.progress(0, text="Starting…")
        status_txt = st.empty()
        results: list[dict] = []
        n_batch = len(selected_df)

        for idx, (_, row) in enumerate(selected_df.iterrows()):
            img_id   = str(row["image"])
            gt       = str(row["label"]).upper().strip()
            img_path = image_dir / img_id

            prog.progress((idx + 1) / n_batch, text=f"{idx+1}/{n_batch}  {img_id}")
            status_txt.caption(f"Processing: **{img_id}**  (GT: {gt})")

            if not img_path.exists():
                results.append({
                    "image_id": img_id, "ground_truth": gt,
                    "final_label": "ERROR",
                    "tally": {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0},
                    "is_correct": False, "judges": [],
                    "error": "image file not found",
                })
                continue

            labels: list[str]       = []
            judge_outputs: list[dict] = []
            judge_metas: list[dict]   = []
            judge_latencies: list[int] = []
            try:
                for cfg in judge_cfgs:
                    t0 = time.time()
                    parsed, jmeta = run_judge(client, cfg, image_path=img_path, receipt_id=img_id)
                    judge_latencies.append(int((time.time() - t0) * 1000))
                    labels.append(parsed["label"])
                    judge_outputs.append({"name": cfg.name, "model": cfg.model, **parsed})
                    judge_metas.append(jmeta)

                final_label = majority_vote(labels)
                tally       = vote_tally(labels)
                is_correct  = final_label == gt

                results.append({
                    "image_id": img_id, "ground_truth": gt,
                    "final_label": final_label, "tally": tally,
                    "is_correct": is_correct, "judges": judge_outputs,
                })

                # ── Comprehensive Langfuse logging ────────────────────────────
                if lf is not None:
                    try:
                        # Image feature analysis (CPU-only, re-used for trace input)
                        try:
                            _stats = image_basic_stats(img_path)
                            _bc    = brightness_contrast(img_path)
                            _blur  = blur_variance_of_laplacian(img_path)
                            _analysis = {
                                **_stats,
                                "brightness_mean": round(_bc["brightness_mean"], 4),
                                "contrast_std":    round(_bc["contrast_std"], 4),
                                "blur_variance":   round(_blur, 2),
                            }
                        except Exception:
                            _analysis = {}

                        # Trace-level input/output — shown in Langfuse dataset-run
                        # comparison view ("Trace Input" / "Output" columns)
                        trace_meta = {
                            "name":    f"batch_eval_{img_id}",
                            "user_id": "streamlit-batch",
                            "input":   {"image_id": img_id, "ground_truth": gt, **_analysis},
                            "output":  {
                                "final_label":  final_label,
                                "tally":        tally,
                                "is_correct":   is_correct,
                                "judge_labels": labels,
                            },
                            "metadata": {
                                "receipt_id":   img_id,
                                "ground_truth": gt,
                                "run":          run_name,
                                "final_label":  final_label,
                                "is_correct":   is_correct,
                            },
                            "tags": ["receipt", "batch-eval", gt.lower(), run_name],
                        }

                        # 1 ── Image analysis span; capture REAL trace ID from response
                        lf_resp = lf.log_observation(
                            name="dataset_analysis",
                            as_type="span",
                            trace=trace_meta,
                            observation={
                                "input":    {"image_id": img_id, "ground_truth": gt},
                                "output":   _analysis,
                                "metadata": {"image_exists": img_path.exists()},
                            },
                        )
                        # Langfuse may assign its own UUID — use it for all subsequent calls
                        actual_trace_id = (
                            lf_resp.get("trace_id")
                            or lf_resp.get("id")
                            or (lf_resp.get("trace_url", "").rsplit("/", 1)[-1]
                                if lf_resp.get("trace_url") else None)
                        )
                        if actual_trace_id:
                            trace_meta = {**trace_meta, "id": actual_trace_id}

                        # 2 ── One generation per judge (full prompt + response + tokens + latency)
                        for cfg, jout, jmeta, lat in zip(
                            judge_cfgs, judge_outputs, judge_metas, judge_latencies
                        ):
                            lf.log_generation(
                                observation={
                                    "name":  cfg.name,
                                    "model": cfg.model,
                                    "input": {
                                        "system": jmeta["input"]["prompt"],
                                        "user":   f"Receipt ID: {img_id}  [image attached]",
                                    },
                                    "output": jmeta["output"]["parsed"],
                                    "usage": {
                                        "input":  jmeta["usage"].get("input_tokens"),
                                        "output": jmeta["usage"].get("output_tokens"),
                                        "total":  jmeta["usage"].get("total_tokens"),
                                        "unit":   "TOKENS",
                                    },
                                    "metadata": {
                                        "temperature":  cfg.temperature,
                                        "persona":      cfg.persona,
                                        "ground_truth": gt,
                                        "latency_ms":   lat,
                                        "raw_response": jmeta["output"]["raw"][:300],
                                    },
                                    "level": "DEFAULT",
                                },
                                trace=trace_meta,
                                latency_ms=lat,
                            )

                        # 3 ── Vote aggregation span
                        lf.log_observation(
                            name="vote_aggregation",
                            as_type="span",
                            trace=trace_meta,
                            observation={
                                "input":    {"judge_labels": labels},
                                "output":   {
                                    "final_label": final_label,
                                    "tally":       tally,
                                    "is_correct":  is_correct,
                                },
                                "metadata": {"ground_truth": gt},
                            },
                        )

                        # 4 ── 8 scores — use actual Langfuse trace ID
                        score_tid = actual_trace_id or ""
                        for cfg, jout in zip(judge_cfgs, judge_outputs):
                            lf.create_score(
                                score_tid, f"{cfg.name}_confidence",
                                float(jout["confidence"]),
                                comment=f"label={jout['label']}",
                            )
                            judge_score = (
                                1.0 if jout["label"] == gt
                                else 0.5 if jout["label"] == "UNCERTAIN"
                                else 0.0
                            )
                            lf.create_score(
                                score_tid, f"{cfg.name}_correctness",
                                judge_score,
                                comment=f"judge={jout['label']} gt={gt}",
                            )
                        lf.create_score(
                            score_tid, "final_correct",
                            1.0 if is_correct else 0.0,
                            comment=f"final={final_label} gt={gt}",
                        )
                        lf.create_score(
                            score_tid, "inter_judge_agreement",
                            1.0 if len(set(labels)) == 1 else 0.0,
                            comment=f"labels={labels}",
                        )

                        # 5 ── Link trace to named dataset run (actual trace ID)
                        lf.dataset_run_log(
                            run_name=run_name,
                            dataset_item_id=img_id,
                            trace_id=score_tid,
                            metadata={
                                "is_correct":   is_correct,
                                "final_label":  final_label,
                                "ground_truth": gt,
                            },
                        )

                    except Exception:
                        pass  # never abort the batch for a Langfuse error

            except Exception as exc:
                results.append({
                    "image_id": img_id, "ground_truth": gt,
                    "final_label": "ERROR",
                    "tally": {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0},
                    "is_correct": False, "judges": [],
                    "error": str(exc),
                })

        prog.empty()
        status_txt.empty()
        st.session_state["batch_results"] = results
        st.rerun()

    # ── Results section ───────────────────────────────────────────────────────
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        n_ok    = sum(1 for r in results if r.get("final_label") != "ERROR")
        n_err   = len(results) - n_ok
        st.success(f"Batch complete — {n_ok} evaluated" + (f", {n_err} errors" if n_err else ""))
        if n_err:
            with st.expander(f"{n_err} errors"):
                st.json([r for r in results if r.get("final_label") == "ERROR"])

        st.divider()
        _batch_results_charts([r for r in results if r.get("final_label") != "ERROR"])
        st.divider()

        st.download_button(
            "⬇  Download full results as JSON",
            data=json.dumps(results, indent=2, default=str),
            file_name="batch_eval_results.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:

    load_dotenv()

    st.title("LLM-Judge Fake Receipt Detector")
    st.caption("3 OpenAI judges + majority vote + Langfuse tracing")

    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    data_mode = st.sidebar.selectbox("Dataset mode", ["sample (included)", "custom path"], index=0)

    if data_mode.startswith("sample"):
        label_path = Path("data/sample/labels.csv")
        image_dir = Path("data/sample/images")
        ocr_dir: Path | None = Path("data/sample/ocr")
    else:
        label_path = Path(_browse_field(
            "labels file (CSV)", "_label_path", "data/findit2/train.txt",
            "b_label", is_dir=False,
            filetypes=[("CSV / TXT files", "*.csv *.txt"), ("All files", "*.*")],
        ))
        image_dir = Path(_browse_field(
            "image dir", "_image_dir", "data/findit2/images",
            "b_imgdir", is_dir=True,
        ))
        ocr_dir_val = _browse_field(
            "ocr dir (optional)", "_ocr_dir", "",
            "b_ocrdir", is_dir=True,
        )
        ocr_dir = Path(ocr_dir_val) if ocr_dir_val else None

    if not label_path.exists():
        st.error(f"Labels file not found: {label_path}")
        st.stop()

    labels_df = pd.read_csv(label_path)
    if "label" not in labels_df.columns:
        if "forged" in labels_df.columns:
            labels_df["label"] = labels_df["forged"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")
        else:
            st.error("Labels file must include either 'label' or 'forged' column.")
            st.stop()

    st.sidebar.write("Rows:", len(labels_df))
    receipt_id = st.sidebar.selectbox("Pick a receipt", labels_df["image"].tolist())

    # ── Langfuse dataset registration ─────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("**Register to Langfuse**")

    lf_name_default = label_path.stem.replace(".", "_")
    lf_dataset_name = st.sidebar.text_input(
        "Dataset name",
        value=lf_name_default,
        help="Langfuse dataset name — created if it does not exist, updated if it does.",
    )

    if st.sidebar.button(" Register dataset items", key="btn_register_lf"):
        try:
            lf = LangfuseMCPClient()
            lf.dataset_create(
                name=lf_dataset_name,
                description=f"Receipts — {label_path.name}  ({len(labels_df)} images)",
                metadata={"source": str(label_path), "n_rows": len(labels_df)},
            )
        except Exception as exc:
            st.sidebar.error(
                f"MCP server not reachable: {exc}\n\n"
                "Make sure the Langfuse MCP server is running at localhost:8005 before registering."
            )
            lf = None

        if lf is not None:
            n      = len(labels_df)
            prog   = st.sidebar.progress(0, text="Starting…")
            status = st.sidebar.empty()
            n_ok   = 0
            errors = []

            for i, (_, row) in enumerate(labels_df.iterrows()):
                img_id   = str(row["image"])
                gt       = str(row["label"]).upper().strip()
                img_path = image_dir / img_id
                ocr_path = (ocr_dir / Path(img_id).with_suffix(".txt").name) if ocr_dir else None

                prog.progress((i + 1) / n, text=f"{i+1}/{n}  {img_id}")

                if not img_path.exists():
                    errors.append(f"{img_id}: image not found")
                    continue

                try:
                    img_stats = image_basic_stats(img_path)
                    bc        = brightness_contrast(img_path)
                    blur      = blur_variance_of_laplacian(img_path)
                    ocr_text  = ocr_path.read_text(errors="ignore") if (ocr_path and ocr_path.exists()) else ""
                    ocr_total = extract_total_from_text(ocr_text) if ocr_text else None

                    meta = {
                        **img_stats,
                        "brightness_mean":     round(bc["brightness_mean"], 4),
                        "contrast_std":        round(bc["contrast_std"], 4),
                        "blur_var_laplacian":  round(blur, 2),
                        "ocr_total_extracted": ocr_total,
                    }

                    lf.dataset_add_item(
                        dataset_name=lf_dataset_name,
                        input={"image_id": img_id, "image_path": str(img_path), "metadata": meta},
                        expected_output={"label": gt},
                        item_id=img_id,
                        metadata={"ground_truth": gt, **meta},
                    )
                    n_ok += 1

                except Exception as exc:
                    errors.append(f"{img_id}: {exc}")

            prog.empty()
            if errors:
                status.warning(f"Done: {n_ok}/{n} registered · {len(errors)} errors")
                with st.sidebar.expander("Errors"):
                    st.write(errors[:20])
            else:
                status.success(f"✓ {n_ok}/{n} images registered in '{lf_dataset_name}'")

    # ── Judge configurations (shared across tabs) ─────────────────────────────
    judge_cfgs = [
        JudgeConfig(name="judge_1", model=settings.judge_models[0], temperature=0.2,
                    persona="strict, skeptical, focuses on forensic inconsistencies"),
        JudgeConfig(name="judge_2", model=settings.judge_models[1], temperature=0.4,
                    persona="balanced, looks for plausible printing/scan artifacts vs tampering"),
        JudgeConfig(name="judge_3", model=settings.judge_models[2], temperature=0.7,
                    persona="lenient, assumes real unless clear signs of manipulation"),
    ]

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Analyze Receipt", "Dataset Statistics", "Batch Evaluation"])

    # ── Tab 1: single-receipt analysis (original functionality) ──────────────
    with tab1:

        row = labels_df[labels_df["image"] == receipt_id].iloc[0].to_dict()
        gt = str(row.get("label", "")).upper().strip() or None

        image_path = image_dir / receipt_id
        if not image_path.exists():
            st.error(f"Image not found: {image_path}")
            st.stop()

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Receipt image")
            st.image(str(image_path), use_container_width=True)

        with c2:
            st.subheader("Extracted fields (lightweight)")
            img_stats = image_basic_stats(image_path)
            img_stats.update(brightness_contrast(image_path))
            img_stats["blur_var_laplacian"] = blur_variance_of_laplacian(image_path)

            total = None
            ocr_text = None
            if ocr_dir is not None:
                ocr_path = ocr_dir / Path(receipt_id).with_suffix(".txt").name
                if ocr_path.exists():
                    ocr_text = ocr_path.read_text(errors="ignore")
                    total = extract_total_from_text(ocr_text)

            st.json({
                "receipt_id": receipt_id,
                "ground_truth": gt,
                "total_amount_parsed": total,
                **img_stats,
            })

        st.divider()
        st.subheader("Run 3 judges")

        run = st.button("Run judges", type="primary")

        if run:
            langfuse = maybe_create_langfuse(
                settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host
            )
            trace = start_trace(
                langfuse,
                name="receipt_streamlit",
                user_id="streamlit-user",
                metadata={"receipt_id": receipt_id, "ground_truth": gt, "total_amount_parsed": total},
                tags=["streamlit", "receipt"],
            )

            outputs = []
            labels = []
            metas = []

            for cfg in judge_cfgs:
                parsed, meta = run_judge(client, cfg, image_path=image_path, receipt_id=receipt_id)
                outputs.append({"judge": cfg.name, "model": cfg.model, **parsed})
                labels.append(parsed["label"])
                metas.append(meta)

                log_generation(
                    trace,
                    name=cfg.name,
                    model=cfg.model,
                    input_payload=meta["input"],
                    output_payload=meta["output"],
                    usage=meta["usage"],
                    metadata={"temperature": cfg.temperature},
                )
                log_score(trace, name=f"{cfg.name}_confidence",
                          value=float(parsed["confidence"]), comment=parsed["label"])

            final_label = majority_vote(labels)
            tally = vote_tally(labels)

            if gt in ("FAKE", "REAL"):
                correct = 1.0 if final_label == gt else 0.0
                log_score(trace, name="final_correct", value=correct,
                          comment=f"final={final_label} gt={gt}")

            flush(langfuse)

            st.success(f"Final: {final_label}  |  Tally: {tally}")

            cols = st.columns(3)
            for i, out in enumerate(outputs):
                with cols[i]:
                    st.markdown(f"### {out['judge']}")
                    st.caption(out["model"])
                    st.json(out)

            with st.expander("Raw outputs (for debugging)"):
                st.json(metas)

            st.caption("If LANGFUSE_* env vars are set, this run is logged as a trace with generations + scores.")

    # ── Tab 2: dataset statistics ─────────────────────────────────────────────
    with tab2:
        _stats_tab(label_path, image_dir, ocr_dir)

    # ── Tab 3: batch evaluation ───────────────────────────────────────────────
    with tab3:
        _batch_tab(labels_df, image_dir, ocr_dir, client, judge_cfgs, settings)


if __name__ == "__main__":
    main()
