from __future__ import annotations

import json
import gc
import tempfile
import sys
from pathlib import Path

# Add repository root explicitly so imports work regardless of launch directory
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd
import cv2
from ultralytics import YOLO
from src.inference.pipeline import CoralPipeline
from src.inference.dinov3_dpt import is_dinov3_dpt

st.set_page_config(
    page_title="AI Coral Reef Health Assessment",
    page_icon="🪸",
    layout="wide",
)

st.title("🪸 AI-Assisted Coral Reef Health Assessment")
st.caption("Automated benthic segmentation, quantitative reef health indicators, and trustworthy AI quality/reliability gating.")

# Sidebar Configuration
st.sidebar.header("Configuration")
model_choices: dict[str, str] = {}
for checkpoint in sorted((REPO_ROOT / "models/checkpoints").glob("**/*.pt")):
    rel_path = checkpoint.relative_to(REPO_ROOT).as_posix()
    model_choices[rel_path] = str(checkpoint)

for config_path in sorted((REPO_ROOT / "models/pretrained").glob("*/config.json")):
    model_dir = config_path.parent
    if any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin")):
        rel_path = model_dir.relative_to(REPO_ROOT).as_posix()
        model_choices[rel_path] = str(model_dir)

if not model_choices:
    st.sidebar.error("No model checkpoints found under models/checkpoints or models/pretrained.")
    st.stop()

preferred_model = "models/checkpoints/yolo26n_sem/weights/best.pt"
default_model = preferred_model if preferred_model in model_choices else next(iter(model_choices))
selected_model_file = st.sidebar.selectbox(
    "Model architecture",
    options=list(model_choices),
    index=list(model_choices).index(default_model),
    help="Select an inference model. SegFormer and DINOv3 are directories; YOLO models are weights files.",
)
model_path = model_choices[selected_model_file]
if Path(model_path).is_dir():
    model_type = "dinov3" if is_dinov3_dpt(model_path) else "segformer"
else:
    model_type = "yolo"


@st.cache_data(show_spinner=False)
def inspect_yolo_checkpoint(checkpoint_path: str, checkpoint_mtime_ns: int) -> dict:
    """Read architecture and training properties saved in an Ultralytics checkpoint."""
    model = YOLO(checkpoint_path)
    checkpoint = getattr(model, "ckpt", None) or {}
    train_args = checkpoint.get("train_args", {}) or {}
    model_yaml = getattr(getattr(model, "model", None), "yaml", {}) or {}
    yaml_file = model_yaml.get("yaml_file")
    model_name = Path(yaml_file).stem if yaml_file else Path(checkpoint_path).parent.parent.name
    names = model.names
    class_names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    try:
        imgsz = int(train_args.get("imgsz", 1024))
    except (TypeError, ValueError):
        imgsz = 1024
    return {
        "name": model_name,
        "task": str(getattr(model, "task", train_args.get("task", "unknown"))),
        "imgsz": imgsz,
        "classes": class_names,
    }


if model_type == "yolo":
    try:
        yolo_properties = inspect_yolo_checkpoint(model_path, Path(model_path).stat().st_mtime_ns)
    except Exception as exc:
        st.sidebar.error(f"Could not read YOLO checkpoint metadata: {exc}")
        st.stop()

if model_type == "dinov3":
    st.sidebar.caption("High-precision foundation model (DINOv3 ViT-L + DPT).")
elif model_type == "segformer":
    st.sidebar.caption("Transformer benchmark (SegFormer-B5).")
else:
    task_label = "semantic segmentation" if yolo_properties["task"] == "semantic" else f"{yolo_properties['task']} model"
    st.sidebar.caption(f"{yolo_properties['name']} · {task_label}")
    st.sidebar.caption(
        f"Saved training size: {yolo_properties['imgsz']}px · "
        f"{len(yolo_properties['classes'])} classes: {', '.join(yolo_properties['classes'])}"
    )

if model_type == "yolo":
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.80, 0.20, 0.05)
    img_size = st.sidebar.select_slider(
        "Image Resolution (px)",
        options=sorted({384, 512, 640, 768, 1024, yolo_properties["imgsz"]}),
        value=yolo_properties["imgsz"],
    )
else:
    conf_thresh = 0.20
    img_size = 1024

st.sidebar.markdown("---")
st.sidebar.markdown("### Ecological Benchmarks (Reef Check)")
st.sidebar.markdown(r"""
- **Excellent**: Live Coral Cover $\ge 50\%$
- **Good**: Live Coral Cover $25\% - 50\%$
- **Fair**: Live Coral Cover $10\% - 25\%$
- **Poor / Degraded**: Live Coral Cover $< 10\%$
""")


@st.cache_resource(max_entries=1)
def load_pipeline(ckpt_path: str, conf: float, imgsz: int, selected_type: str):
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return CoralPipeline(
        model_path=ckpt_path,
        imgsz=imgsz,
        conf=conf,
        model_type=selected_type,
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

                    pipeline = load_pipeline(model_path, conf_thresh, img_size, model_type)
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
                        st.metric("Algae-Covered Area", f"{health.get('algae_covered_cover_pct', health['macroalgae_cover_pct'])}%",
                                  delta=f"Coral/Algae-Covered: {health['coral_to_algae_ratio']}",
                                  delta_color="normal",
                                  help="Area annotated as algae-covered in the segmentation labels.")
                    with col4:
                        st.metric("Reef Health Status", health["reef_health_category"].split(" [")[0],
                                  help=health["reef_health_category"])
                        st.caption(
                            f"{health.get('coral_condition', 'Stable')} · "
                            f"cover tier {health.get('live_cover_tier', 'n/a')}"
                        )

                    st.markdown("---")

                    # 3. Image Comparison
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.subheader("Input Imagery")
                        st.image(str(image_path), width="stretch")
                    with img_col2:
                        st.subheader("Benthic Segmentation Overlay")
                        st.image(str(overlay_path), width="stretch")

                    # 4. Details & Composition Table
                    tab1, tab2 = st.tabs(["📊 Benthic Composition", "🛡️ Quality & Trust Signals"])

                    with tab1:
                        st.markdown("#### Benthic Area Breakdown")
                        comp = report["benthic_composition"]
                        comp_df = pd.DataFrame([
                            {"Class": k.replace("_", " ").title(), "Area Fraction (%)": round(v * 100, 2)}
                            for k, v in comp.items()
                        ])
                        st.dataframe(comp_df, width="stretch")

                    with tab2:
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
                            conf = report.get("mean_confidence")
                            unc = report.get("prediction_uncertainty")
                            st.write(f"- **Mean Prediction Confidence**: {f'{conf:.3f}' if conf is not None else 'n/a (backend returns class IDs only)'}")
                            st.write(f"- **Prediction Uncertainty Score**: {f'{unc:.3f}' if unc is not None else 'n/a'}")
                            st.write(f"- **Review Triggers**: {rel['reasons'] if rel['reasons'] else 'None'}")

                    st.download_button(
                        "📥 Download Health Report (JSON)",
                        json.dumps(report, indent=2),
                        file_name=f"reef_health_report_{Path(uploaded.name).stem}.json",
                        mime="application/json",
                    )

else:
    uploaded_video = st.file_uploader("Upload an underwater survey video", type=["mp4", "avi", "mov", "mkv"])
    vcol_a, vcol_b = st.columns(2)
    with vcol_a:
        every_n = st.slider("Process Every N Frames (1 = full video)", 1, 60, 2,
                            help="1 processes every frame. 2 or 5 processes every 2nd/5th frame for higher throughput.")
    with vcol_b:
        process_entire = st.checkbox("Process Entire Video (All Frames)", value=True,
                                     help="When checked, processes the complete video from start to finish without truncation.")
        if not process_entire:
            max_frames = st.number_input("Max frames to process", min_value=10, max_value=200000, value=300, step=50,
                                         help="Caps total frames processed for quick previews.")
        else:
            max_frames = None

    if uploaded_video:
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: {model_path}.")
        else:
            progress_bar = st.progress(0, text="Initializing video inference...")
            with tempfile.TemporaryDirectory() as tmp:
                video_path = Path(tmp) / uploaded_video.name
                video_path.write_bytes(uploaded_video.getbuffer())
                out_dir = Path(tmp) / "video_out"

                pipeline = load_pipeline(model_path, conf_thresh, img_size, model_type)

                def _progress_cb(current: int, total: int):
                    frac = min(1.0, current / max(1, total))
                    progress_bar.progress(frac, text=f"Processing frame {current} of ~{total} ({int(frac * 100)}%)...")

                result = pipeline.infer_video(
                    video_path, out_dir, every_n=every_n, max_frames=max_frames,
                    save_frames=True, progress_callback=_progress_cb
                )
                progress_bar.progress(1.0, text="Finalizing video encoding...")
                summary = result["summary"]

                video_output_bytes = None
                if result.get("annotated_video") and Path(result["annotated_video"]).exists():
                    video_output_bytes = Path(result["annotated_video"]).read_bytes()

                overlays = sorted(out_dir.glob("*_overlay.jpg"))[:6]
                overlay_images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in overlays if p.exists()]
            progress_bar.empty()

            st.subheader(f"Survey Assessment: {summary.get('overall_health_category', 'Assessed')}")
            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.metric("Accepted Frames", f"{summary['accepted_frames']} / {summary['total_frames']}")
            with vcol2:
                st.metric("Survey Mean Live Coral Cover", f"{summary['aggregated_health_metrics']['mean_live_coral_cover_pct']}%")
            with vcol3:
                st.metric("Survey Mean Bleaching Ratio", f"{summary['aggregated_health_metrics']['mean_bleaching_ratio_pct']}%")

            if video_output_bytes:
                st.subheader("Annotated Video Stream")
                st.video(video_output_bytes)
                st.download_button(
                    "📥 Download Annotated Video (.mp4)",
                    data=video_output_bytes,
                    file_name=f"annotated_{Path(uploaded_video.name).stem}.mp4",
                    mime="video/mp4",
                )

            if overlay_images:
                st.subheader("Sampled Keyframe Overlays")
                st.image(overlay_images, caption=[p.name for p in overlays], width=350)

            st.download_button(
                "📥 Download Video Survey Report (JSON)",
                json.dumps(result, indent=2),
                file_name="video_survey_report.json",
                mime="application/json",
            )
