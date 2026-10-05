from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

# Streamlit executes this file as a script. Add the repository root explicitly
# so `from src...` works even when the command is launched outside the repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from src.inference.pipeline import CoralPipeline

st.set_page_config(page_title="Trustworthy Coral Reef Assessment", layout="wide")
st.title("Trustworthy AI-Assisted Coral Reef Health Assessment")
st.caption("Visible benthic and coral-condition indicators with explicit quality and review warnings.")
model_path = st.sidebar.text_input("Model checkpoint", "models/checkpoints/yolo26n_seg/weights/best.pt")
mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

if mode == "Image":
    uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])
    if uploaded:
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: {model_path}. Run the training command first.")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                image_path, overlay_path = Path(tmp) / uploaded.name, Path(tmp) / "overlay.jpg"
                image_path.write_bytes(uploaded.getbuffer()); report = CoralPipeline(model_path).infer(image_path, overlay_path)
                st.image([str(image_path), str(overlay_path)], caption=["Input", "Model overlay"], width=500)
                status = report["reliability"]["status"]
                (st.success if status == "accepted" else st.warning)(f"Result status: {status}")
                st.json(report)
                st.download_button("Download JSON report", json.dumps(report, indent=2), "coral_report.json", "application/json")
else:
    uploaded = st.file_uploader("Upload an underwater video", type=["mp4", "avi", "mov", "mkv"])
    every_n = st.slider("Sample every N frames", 5, 60, 15)
    max_frames = st.slider("Max sampled frames", 4, 60, 12)
    if uploaded:
        if not Path(model_path).exists():
            st.error(f"Checkpoint not found: {model_path}. Run the training command first.")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                video_path = Path(tmp) / uploaded.name
                video_path.write_bytes(uploaded.getbuffer())
                out_dir = Path(tmp) / "video_out"
                with st.spinner("Sampling frames and running segmentation..."):
                    report = CoralPipeline(model_path).infer_video(video_path, out_dir, every_n=every_n, max_frames=max_frames)
                summary = report["summary"]
                status = summary["status"]
                (st.success if status == "accepted" else st.warning)(
                    f"Video status: {status} — {summary['accepted_frames']}/{summary['total_frames']} frames accepted")
                st.json(summary)
                overlays = sorted(out_dir.glob("*_overlay.jpg"))[:6]
                if overlays:
                    st.image([str(p) for p in overlays], caption=[p.name for p in overlays], width=350)
                st.json(report)
                st.download_button("Download JSON report", json.dumps(report, indent=2), "coral_video_report.json", "application/json")
