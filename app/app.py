from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st

# Allow app/ to import from project root src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.tracking.tracker import PlayerTracker
from src.tracking.postprocess import clean_tracking_data
from src.team_assignment.cluster import (
    compute_average_positions,
    cluster_teams,
    save_team_assignments,
    plot_team_clusters,
)
from src.formation.formation_detector import detect_formations


UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "outputs" / "streamlit"


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded Streamlit video to disk.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    video_path = UPLOAD_DIR / uploaded_file.name

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return video_path


def convert_to_browser_mp4(input_path: Path, output_path: Path) -> Path:
    """
    Convert OpenCV-generated MP4 into browser-compatible H264 MP4.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )

    return output_path


def run_analysis(video_path: Path) -> dict:
    """
    Full tactical analysis pipeline.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Outputs
    raw_video_out = OUTPUT_DIR / "tracked_video_raw.mp4"
    browser_video_out = OUTPUT_DIR / "tracked_video_browser.mp4"

    raw_csv = OUTPUT_DIR / "tracking_raw.csv"
    clean_csv = OUTPUT_DIR / "tracking_clean.csv"

    avg_csv = OUTPUT_DIR / "average_positions.csv"

    team_csv = OUTPUT_DIR / "team_clusters.csv"
    team_plot = OUTPUT_DIR / "team_clusters.png"

   
    # Tracking
    tracker = PlayerTracker(
        model_path="yolov8s.pt",
        conf_thresh=0.25,
        iou_thresh=0.5,
        imgsz=960,
        tracker_config="bytetrack.yaml",
        frame_skip=1,
        filter_to_pitch=True,
        pitch_erode_px=5,
    )

    tracker.run(
        video_in=video_path,
        video_out=raw_video_out,
        csv_out=raw_csv,
    )

    # Convert for browser playback
    convert_to_browser_mp4(raw_video_out, browser_video_out)

  
    # Postprocess tracking

    clean_tracking_data(
        csv_in=raw_csv,
        csv_out=clean_csv,
        min_conf=0.25,
        min_track_length=5,
    )

    
    # Compute average positions
    
    player_positions = compute_average_positions(
        csv_in=clean_csv,
        csv_out=avg_csv,
        min_samples_per_track=20,
        max_tracks=24,
    )

    
    # Team clustering
   
    clustered = cluster_teams(
        player_positions,
        tracking_csv=clean_csv,
        video_in=video_path,
    )

    save_team_assignments(clustered, team_csv)

    plot_team_clusters(clustered, team_plot)

    # Formation detection
    
    formation_results = detect_formations(
        clustered_df=clustered,
        k=3,
        max_dist=350,
        reverse_team_1=True,
    )

    return {
        "tracked_video": browser_video_out,
        "team_plot": team_plot,
        "team_csv": team_csv,
        "team0_formation": formation_results["team0_formation"],
        "team1_formation": formation_results["team1_formation"],
    }

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Football Tactical Analysis",
    page_icon="⚽",
    layout="wide",
)

# -----------------------------
# Global CSS
# -----------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617, #0f172a);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .landing-container {
        height: 85vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .landing-title {
        font-size: 4rem;
        font-weight: 900;
        color: #f8fafc;
        margin-bottom: 1rem;
    }

    .landing-subtitle {
        font-size: 1.25rem;
        color: #cbd5e1;
        max-width: 850px;
        margin-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #14532d, #0f172a);
        padding: 2rem;
        border-radius: 22px;
        margin-bottom: 2rem;
        border: 1px solid #22c55e;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        max-width: 850px;
    }

    .formation-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
    }

    .formation-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-bottom: 0.4rem;
    }

    .formation-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #22c55e;
    }

    div[data-testid="stFileUploader"] {
        background-color: #111827;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #334155;
    }

    .stButton > button {
        background-color: #22c55e;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #16a34a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


# -----------------------------
# Home / title page
# -----------------------------

def show_home():
    st.markdown(
        """
        <div class="landing-container">
            <div class="landing-title">⚽ Football Tactical Analysis</div>
            <div class="landing-subtitle">
                Automated player tracking, team clustering and formation detection
                from broadcast football footage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("▶ Enter Analysis Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


# -----------------------------
# Dashboard
# -----------------------------

def show_dashboard():
    with st.sidebar:
        st.header("System Configuration")

        st.markdown("**Detection/Tracking:** YOLOv8s + ByteTrack")
        st.markdown("**Confidence Threshold:** 0.25")
        st.markdown("**Image Size:** 960")
        st.markdown("**Formation Lines:** 3")
        st.markdown("**Max Tracks:** 24")

        st.divider()

        st.info(
            "These settings were selected during development to balance "
            "tracking coverage, speed and formation stability."
        )

        if st.button("← Back to Title Page"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Football Tactical Analysis Dashboard</div>
            <div class="hero-subtitle">
                Upload a short match clip to generate tracking, team clustering
                and formation outputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a football clip",
        type=["mp4", "mkv", "mov"],
    )

    if uploaded_file is None:
        st.info("Upload a short football clip to begin tactical analysis.")
        return

    video_path = save_uploaded_file(uploaded_file)

    st.markdown("## Uploaded Clip")
    st.video(str(video_path))

    if st.button("Run Tactical Analysis", use_container_width=True):
        with st.spinner("Running player tracking, team clustering and formation detection..."):
            results = run_analysis(video_path)

        st.success("Analysis complete")

        st.markdown("## Results Overview")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                f"""
                <div class="formation-card">
                    <div class="formation-label">Team 0 Formation</div>
                    <div class="formation-value">{results["team0_formation"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                f"""
                <div class="formation-card">
                    <div class="formation-label">Team 1 Formation</div>
                    <div class="formation-value">{results["team1_formation"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        tab1, tab2, tab3 = st.tabs(
            ["Tracked Video", "Team Clustering", "Export Data"]
        )

        with tab1:
            st.markdown("### Player Tracking Output")
            st.caption("Annotated video showing detected players and tracking IDs.")
            st.video(str(results["tracked_video"]))

        with tab2:
            st.markdown("### Team Clustering Output")
            st.caption("Average tracked player positions clustered into two teams.")
            st.image(str(results["team_plot"]), use_container_width=True)

        with tab3:
            st.markdown("### Download Results")
            st.caption("Download the generated team clustering data.")

            st.download_button(
                label="Download Team Cluster CSV",
                data=Path(results["team_csv"]).read_bytes(),
                file_name="team_clusters.csv",
                mime="text/csv",
            )


# -----------------------------
# Page routing
# -----------------------------

if st.session_state.page == "home":
    show_home()
else:
    show_dashboard()
