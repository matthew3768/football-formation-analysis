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
   
    clustered = cluster_teams(player_positions)

    save_team_assignments(clustered, team_csv)

    plot_team_clusters(clustered, team_plot)

    # Formation detection
    
    formation_results = detect_formations(
        clustered_df=clustered,
        k=3,
        max_dist=250,
        reverse_team_1=True,
    )

    return {
        "tracked_video": browser_video_out,
        "team_plot": team_plot,
        "team_csv": team_csv,
        "team0_formation": formation_results["team0_formation"],
        "team1_formation": formation_results["team1_formation"],
    }

# Streamlit UI

st.set_page_config(
    page_title="Football Tactical Analysis",
    layout="wide",
)

st.title("Football Tactical Analysis System")

st.write(
    """
Upload a football clip to:
- detect players
- track movement
- cluster teams
- estimate tactical formations
"""
)

uploaded_file = st.file_uploader(
    "Upload football video",
    type=["mp4", "mkv", "mov"],
)

if uploaded_file is not None:

    video_path = save_uploaded_file(uploaded_file)

    st.subheader("Uploaded Video")

    st.video(str(video_path))

    if st.button("Run Analysis"):

        with st.spinner("Running tactical analysis pipeline..."):

            results = run_analysis(video_path)

        st.success("Analysis complete")

        
        # Main outputs
    
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tracked Video")

            st.video(str(results["tracked_video"]))

        with col2:
            st.subheader("Team Clustering")

            st.image(
                str(results["team_plot"]),
                use_container_width=True,
            )

        # Formations

        st.subheader("Detected Formations")

        st.write(f"### Team 0 Formation: `{results['team0_formation']}`")

        st.write(f"### Team 1 Formation: `{results['team1_formation']}`")

      
        # Download CSV
        
        st.download_button(
            label="Download Team Cluster CSV",
            data=Path(results["team_csv"]).read_bytes(),
            file_name="team_clusters.csv",
            mime="text/csv",
        )