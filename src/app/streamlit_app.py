from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

# Add repository root explicitly so imports work regardless of launch directory
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd
from src.inference.pipeline import CoralPipeline

st.set_page_config(
    page_title="AI Coral Reef Health Assessment",
    page_icon="🪸",
    layout="wide",
)

st.title("🪸 AI-Assisted Coral Reef Health Assessment")
st.caption("Automated benthic segmentation, quantitative reef health indicators, and trustworthy AI quality/reliability gating.")

# Sidebar Configuration
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Segmentation Model Checkpoint", "models/checkpoints/yolo26n_seg/weights/best.pt")
condition_model_path = st.sidebar.text_input("Condition Classifier Checkpoint", "models/checkpoints/condition_resnet18.pt")
conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.05, 0.80, 0.20, 0.05)
img_size = st.sidebar.select_slider("Image Resolution (px)", options=[384, 512, 640], value=512)

st.sidebar.markdown("---")
st.sidebar.markdown("### Ecological Benchmarks (Reef Check)")
st.sidebar.markdown(r"""
- **Excellent**: Live Coral Cover $\ge 50\%$
- **Good**: Live Coral Cover $25\% - 50\%$
- **Fair**: Live Coral Cover $10\% - 25\%$
- **Poor / Degraded**: Live Coral Cover $< 10\%$
""")


@st.cache_resource
def load_pipeline(ckpt_path: str, cond_path: str, conf: float, imgsz: int):
    return CoralPipeline(
        model_path=ckpt_path,
        condition_model_path=cond_path if Path(cond_path).exists() else None,
        imgsz=imgsz,
        conf=conf,
    )


mode = st.radio("Input Modality", ["Single Image", "Underwater Video"], horizontal=True)

if mode == "Single Image":
    uploaded = st.file_uploader("Upload an underwater reef image", type=["jpg", "jpeg", "png"])
    if uploaded:
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: {model_path}. Please train or download model weights.")
        else:
            with st.spinner("Analyzing benthic composition and reef health..."):
                with tempfile.TemporaryDirectory() as tmp:
                    image_path = Path(tmp) / uploaded.name
                    overlay_path = Path(tmp) / "overlay.jpg"
                    image_path.write_bytes(uploaded.getbuffer())

                    pipeline = load_pipeline(model_path, condition_model_path, conf_thresh, img_size)
                    report = pipeline.infer(image_path, overlay_path)

                    # 1. Trust & Reliability Banner
                    rel = report["reliability"]
                    qual = report["quality"]
                    health = report["health_indicators"]

                    if rel["status"] == "accepted":
                        st.success(f"✅ **Trust Status: ACCEPTED** — Image quality and model confidence satisfy reliability criteria.")
                    else:
                        st.warning(f"⚠️ **Trust Status: NEEDS EXPERT REVIEW** — System flagged review triggers: `{', '.join(rel['reasons'])}`")

                    # 2. Main Metrics Display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Live Coral Cover (LCC)", f"{health['live_coral_cover_pct']}%",
                                  help="Percentage of benthic area occupied by living coral colonies.")
                    with col2:
                        st.metric("Bleaching Prevalence", f"{health['bleaching_ratio_pct']}%",
                                  delta=f"{health['bleached_coral_cover_pct']}% benthic cover",
                                  delta_color="inverse",
                                  help="Ratio of bleached coral to total coral colony area.")
                    with col3:
                        st.metric("Macroalgae Cover", f"{health['macroalgae_cover_pct']}%",
                                  delta=f"Coral/Algae: {health['coral_to_algae_ratio']}",
                                  delta_color="normal",
                                  help="Algal cover indicates potential macroalgal competition.")
                    with col4:
                        st.metric("Reef Health Status", health["reef_health_category"])

                    st.markdown("---")

                    # 3. Image Comparison
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.subheader("Input Imagery")
                        st.image(str(image_path), use_container_width=True)
                    with img_col2:
                        st.subheader("Benthic Segmentation Overlay")
                        st.image(str(overlay_path), use_container_width=True)

                    # 4. Details & Composition Table
                    tab1, tab2, tab3 = st.tabs(["📊 Benthic Composition", "🔬 Crop Condition Analysis", "🛡️ Quality & Trust Signals"])

                    with tab1:
                        st.markdown("#### Benthic Area Breakdown")
                        comp = report["benthic_composition"]
                        comp_df = pd.DataFrame([
                            {"Class": k.replace("_", " ").title(), "Area Fraction (%)": round(v * 100, 2)}
                            for k, v in comp.items()
                        ])
                        st.dataframe(comp_df, use_container_width=True)

                    with tab2:
                        if report["condition_classifier_evaluated"]:
                            st.markdown("#### Crop-Level Coral Condition Verification")
                            crops = report["crop_evaluations"]
                            st.info(f"Evaluated {len(crops)} coral crops using secondary condition classifier.")
                            crop_df = pd.DataFrame([
                                {
                                    "Detection ID": c["detection_id"],
                                    "Predicted Condition": c["predicted_condition"],
                                    "Alive Probability": round(c["probabilities"].get("coral_alive", 0), 3),
                                    "Bleached Probability": round(c["probabilities"].get("coral_bleached", 0), 3),
                                    "Dead Probability": round(c["probabilities"].get("coral_dead", 0), 3),
                                }
                                for c in crops
                            ])
                            st.dataframe(crop_df, use_container_width=True)
                        else:
                            st.caption("Condition was directly estimated from YOLO multi-class segmentation masks.")

                    with tab3:
                        st.markdown("#### Image Quality & Uncertainty Signals")
                        qcol1, qcol2 = st.columns(2)
                        with qcol1:
                            st.write(f"- **Blur Variance**: {qual.get('blur_score', 0):.2f}")
                            st.write(f"- **Mean Brightness**: {qual.get('brightness_mean', 0):.2f}")
                            st.write(f"- **Contrast (Std)**: {qual.get('contrast_std', 0):.2f}")
                            st.write(f"- **Color Cast Detected**: {qual.get('color_cast', False)}")
                            st.write(f"- **Overall Quality Score**: {qual.get('quality_score', 0):.2f}")
                        with qcol2:
                            st.write(f"- **Segmented Benthic Coverage**: {report['coverage']*100:.1f}%")
                            st.write(f"- **Mean Detection Confidence**: {report['mean_confidence']:.3f}")
                            st.write(f"- **Prediction Uncertainty Score**: {report['prediction_uncertainty']:.3f}")
                            st.write(f"- **Review Triggers**: {rel['reasons'] if rel['reasons'] else 'None'}")

                    st.download_button(
                        "📥 Download Health Report (JSON)",
                        json.dumps(report, indent=2),
                        file_name=f"reef_health_report_{Path(uploaded.name).stem}.json",
                        mime="application/json",
                    )

else:
    uploaded_video = st.file_uploader("Upload an underwater survey video", type=["mp4", "avi", "mov", "mkv"])
    every_n = st.slider("Sample Every N Frames", 5, 60, 15)
    max_frames = st.slider("Maximum Sampled Frames", 4, 60, 12)

    if uploaded_video:
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: {model_path}.")
        else:
            with st.spinner("Extracting survey frames and assessing reef health across video..."):
                with tempfile.TemporaryDirectory() as tmp:
                    video_path = Path(tmp) / uploaded_video.name
                    video_path.write_bytes(uploaded_video.getbuffer())
                    out_dir = Path(tmp) / "video_out"

                    pipeline = load_pipeline(model_path, condition_model_path, conf_thresh, img_size)
                    result = pipeline.infer_video(video_path, out_dir, every_n=every_n, max_frames=max_frames)
                    summary = result["summary"]

                    st.subheader(f"Survey Assessment: {summary.get('overall_health_category', 'Assessed')}")
                    vcol1, vcol2, vcol3 = st.columns(3)
                    with vcol1:
                        st.metric("Accepted Frames", f"{summary['accepted_frames']} / {summary['total_frames']}")
                    with vcol2:
                        st.metric("Survey Mean Live Coral Cover", f"{summary['aggregated_health_metrics']['mean_live_coral_cover_pct']}%")
                    with vcol3:
                        st.metric("Survey Mean Bleaching Ratio", f"{summary['aggregated_health_metrics']['mean_bleaching_ratio_pct']}%")

                    overlays = sorted(out_dir.glob("*_overlay.jpg"))[:6]
                    if overlays:
                        st.subheader("Sampled Video Overlays")
                        st.image([str(p) for p in overlays], caption=[p.name for p in overlays], width=350)

                    st.download_button(
                        "📥 Download Video Survey Report (JSON)",
                        json.dumps(result, indent=2),
                        file_name="video_survey_report.json",
                        mime="application/json",
                    )
