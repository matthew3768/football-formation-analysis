from pathlib import Path
import json
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st

# Allow app/ to import from project root src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.tracking.tracker import PlayerTracker
from src.tracking.postprocess import clean_tracking_data, select_stable_window
from src.team_assignment.cluster import (
    compute_average_positions,
    cluster_teams,
    keep_best_players_per_team,
    save_team_assignments,
    plot_team_clusters,
)
from src.formation.formation_detector import detect_formations
from src.formation.phase_analysis import (
    average_positions_for_tracks,
    collect_phase_windows,
    detect_ball_positions,
    estimate_possession,
    select_phase_window,
    smooth_possession,
)


UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "outputs" / "streamlit"


def describe_team_colour(hex_colour: str) -> str:
    """
    Convert learned team hex colours into simple user-facing labels.
    """
    hex_colour = hex_colour.lstrip("#")
    r, g, b = tuple(int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))

    if r > b and r > g:
        return "Red Team"
    if b > r and b > g:
        return "Blue Team"
    if g > r and g > b:
        return "Green Team"
    return "Team"


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


def run_analysis(
    video_path: Path,
    tracking_progress_callback=None,
    stage_callback=None,
) -> dict:
    """
    Full tactical analysis pipeline.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # File outputs used across the pipeline and the dashboard.
    raw_video_out = OUTPUT_DIR / "tracked_video_raw.mp4"
    browser_video_out = OUTPUT_DIR / "tracked_video_browser.mp4"

    raw_csv = OUTPUT_DIR / "tracking_raw.csv"
    clean_csv = OUTPUT_DIR / "tracking_clean.csv"

    avg_csv = OUTPUT_DIR / "average_positions.csv"

    team_csv = OUTPUT_DIR / "team_clusters.csv"
    team_plot = OUTPUT_DIR / "team_clusters.png"
    phase_summary_json = OUTPUT_DIR / "phase_analysis_summary.json"
    phase_windows_csv = OUTPUT_DIR / "phase_analysis_windows.csv"

   
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
        progress_callback=tracking_progress_callback,
    )

    if stage_callback is not None:
        stage_callback("Converting tracked video for browser playback...")
    convert_to_browser_mp4(raw_video_out, browser_video_out)

    if stage_callback is not None:
        stage_callback("Cleaning tracking data...")

    clean_tracking_data(
        csv_in=raw_csv,
        csv_out=clean_csv,
        min_conf=0.25,
        min_track_length=5,
    )

    if stage_callback is not None:
        stage_callback("Selecting the most stable analysis window...")
    stable_window = select_stable_window(
        csv_in=clean_csv,
        window_size=250,
        step_size=25,
        min_tracks=18,
    )

    if stage_callback is not None:
        stage_callback("Computing average player positions...")
    
    player_positions = compute_average_positions(
        csv_in=clean_csv,
        csv_out=avg_csv,
        min_samples_per_track=20,
        max_tracks=24,
        frame_range=stable_window,
    )

    if stage_callback is not None:
        stage_callback("Assigning teams from kit colours...")
   
    clustered = cluster_teams(
        player_positions,
        tracking_csv=clean_csv,
        video_in=video_path,
    )
    clustered = keep_best_players_per_team(clustered, max_players_per_team=11)

    save_team_assignments(clustered, team_csv)

    plot_team_clusters(clustered, team_plot)

    if stage_callback is not None:
        stage_callback("Detecting overall observed formations...")
    
    formation_results = detect_formations(
        clustered_df=clustered,
        k=3,
        max_dist=350,
        reverse_team_1=True,
    )

    team_labels = {
        int(team_id): describe_team_colour(team_colour)
        for team_id, team_colour in clustered.groupby("team")["team_colour"].first().items()
    }

    # Phase summaries aggregate multiple valid windows rather than relying on one moment.
    phase_results = {}
    phase_window_rows = []
    if stage_callback is not None:
        stage_callback("Detecting the ball for phase analysis...")
    ball_positions = detect_ball_positions(video_path)
    if stage_callback is not None:
        stage_callback("Estimating and smoothing possession...")
    possession_df = estimate_possession(
        tracking_csv=clean_csv,
        team_assignments=clustered,
        ball_positions=ball_positions,
    )
    smoothed_possession_df = smooth_possession(possession_df)

    if stage_callback is not None:
        stage_callback("Building attacking and defending shape summaries...")
    for team_id in [0, 1]:
        attack_window = select_phase_window(
            tracking_csv=clean_csv,
            possession_df=smoothed_possession_df,
            target_team=team_id,
            analysis_team=team_id,
            team_assignments=clustered,
        )
        defend_window = select_phase_window(
            tracking_csv=clean_csv,
            possession_df=smoothed_possession_df,
            target_team=1 - team_id,
            analysis_team=team_id,
            team_assignments=clustered,
        )

        team_phase_result = {
            "attack_window": attack_window,
            "defend_window": defend_window,
            "attack_shape": None,
            "defend_shape": None,
            "attack_summary": None,
            "defend_summary": None,
        }

        if attack_window is not None:
            attack_positions = average_positions_for_tracks(
                tracking_csv=clean_csv,
                team_assignments=clustered,
                frame_range=attack_window,
            )
            visible_attack_players = int(
                (attack_positions["team"] == team_id).sum()
            )
            if visible_attack_players >= 11:
                try:
                    attack_formations = detect_formations(
                        clustered_df=attack_positions,
                        k=3,
                        max_dist=350,
                        reverse_team_1=True,
                    )
                    team_phase_result["attack_shape"] = attack_formations[
                        f"team{team_id}_formation"
                    ]
                except ValueError:
                    pass

        if defend_window is not None:
            defend_positions = average_positions_for_tracks(
                tracking_csv=clean_csv,
                team_assignments=clustered,
                frame_range=defend_window,
            )
            visible_defend_players = int(
                (defend_positions["team"] == team_id).sum()
            )
            if visible_defend_players >= 11:
                try:
                    defend_formations = detect_formations(
                        clustered_df=defend_positions,
                        k=3,
                        max_dist=350,
                        reverse_team_1=True,
                    )
                    team_phase_result["defend_shape"] = defend_formations[
                        f"team{team_id}_formation"
                    ]
                except ValueError:
                    pass

        phase_results[team_id] = team_phase_result

        for phase_name, target_team in [
            ("attack", team_id),
            ("defend", 1 - team_id),
        ]:
            candidate_windows = collect_phase_windows(
                tracking_csv=clean_csv,
                possession_df=smoothed_possession_df,
                target_team=target_team,
                analysis_team=team_id,
                team_assignments=clustered,
            )

            votes = []
            for candidate in candidate_windows:
                phase_positions = average_positions_for_tracks(
                    tracking_csv=clean_csv,
                    team_assignments=clustered,
                    frame_range=candidate["window"],
                )
                visible_players = int((phase_positions["team"] == team_id).sum())
                if visible_players < 11:
                    continue

                try:
                    phase_formations = detect_formations(
                        clustered_df=phase_positions,
                        k=3,
                        max_dist=350,
                        reverse_team_1=True,
                    )
                except ValueError:
                    continue

                votes.append(
                    {
                        "window": candidate["window"],
                        "formation": phase_formations[f"team{team_id}_formation"],
                        "runner_up": phase_formations[
                            f"team{team_id}_diagnostics"
                        ]["runner_up"],
                        "score_gap": phase_formations[
                            f"team{team_id}_diagnostics"
                        ]["score_gap"],
                        "score": candidate["score"],
                    }
                )
                phase_window_rows.append(
                    {
                        "team": team_id,
                        "phase": phase_name,
                        "window_start": candidate["window"][0],
                        "window_end": candidate["window"][1],
                        "formation": phase_formations[f"team{team_id}_formation"],
                        "runner_up": (
                            phase_formations[f"team{team_id}_diagnostics"]["runner_up"][
                                "formation"
                            ]
                            if phase_formations[f"team{team_id}_diagnostics"]["runner_up"]
                            is not None
                            else None
                        ),
                        "score_gap": phase_formations[
                            f"team{team_id}_diagnostics"
                        ]["score_gap"],
                        "window_score": candidate["score"],
                    }
                )

            if votes:
                vote_counts = pd.Series([vote["formation"] for vote in votes]).value_counts()
                winning_shape = vote_counts.index[0]
                team_phase_result[f"{phase_name}_summary"] = {
                    "shape": winning_shape,
                    "votes": int(vote_counts.iloc[0]),
                    "total_windows": int(len(votes)),
                    "distribution": vote_counts.to_dict(),
                    "runner_ups": [
                        vote["runner_up"]
                        for vote in votes
                        if vote["formation"] == winning_shape
                        and vote["runner_up"] is not None
                    ],
                }

    phase_summary_payload = {
        "team_labels": team_labels,
        "ball_detections": len(ball_positions),
        "possession_observations": len(possession_df),
        "smoothed_possession_observations": len(smoothed_possession_df),
        "phase_results": phase_results,
    }
    phase_summary_json.write_text(
        json.dumps(phase_summary_payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(phase_window_rows).to_csv(phase_windows_csv, index=False)

    return {
        "tracked_video": browser_video_out,
        "team_plot": team_plot,
        "team_csv": team_csv,
        "team0_formation": formation_results["team0_formation"],
        "team1_formation": formation_results["team1_formation"],
        "team0_diagnostics": formation_results["team0_diagnostics"],
        "team1_diagnostics": formation_results["team1_diagnostics"],
        "stable_window": stable_window,
        "phase_results": phase_results,
        "ball_detections": len(ball_positions),
        "possession_observations": len(possession_df),
        "smoothed_possession_observations": len(smoothed_possession_df),
        "team_labels": team_labels,
        "phase_summary_json": phase_summary_json,
        "phase_windows_csv": phase_windows_csv,
    }

# Streamlit UI

st.set_page_config(
    page_title="Football Tactical Analysis",
    page_icon="⚽",
    layout="wide",
)

# Global CSS

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
        height: 58vh;
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
        margin-bottom: 1rem;
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

    .formation-runner {
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #cbd5e1;
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

# Session state

if "page" not in st.session_state:
    st.session_state.page = "home"


# Home / title page

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


# Dashboard

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
        tracking_progress = st.progress(0, text="Tracking players: 0%")
        status_box = st.empty()

        def update_tracking_progress(progress: float, frame_idx: int, total_frames: int) -> None:
            percent = int(progress * 100)
            tracking_progress.progress(
                min(percent, 100),
                text=f"Tracking players: {percent}% ({frame_idx}/{total_frames} frames)",
            )

        with st.spinner("Running player tracking, team clustering and formation detection..."):
            status_box.info("Tracking players across the clip...")
            results = run_analysis(
                video_path,
                tracking_progress_callback=update_tracking_progress,
                stage_callback=lambda message: status_box.info(message),
            )

        tracking_progress.progress(100, text="Tracking complete")
        status_box.success("Tracking, clustering and formation analysis complete.")

        st.success("Analysis complete")

        st.markdown("## Results Overview")

        col_a, col_b = st.columns(2)

        def runner_up_markup(team_key: str) -> str:
            diagnostics = results[f"{team_key}_diagnostics"]
            runner = diagnostics["runner_up"]
            if runner is None:
                return ""
            return (
                '<div class="formation-runner">'
                f"Runner-up: {runner['formation']}"
                "</div>"
            )

        with col_a:
            st.markdown(
                f"""
                <div class="formation-card">
                    <div class="formation-label">{results["team_labels"].get(0, "Team 0")} Formation</div>
                    <div class="formation-value">{results["team0_formation"]}</div>
                    {runner_up_markup("team0")}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                f"""
                <div class="formation-card">
                    <div class="formation-label">{results["team_labels"].get(1, "Team 1")} Formation</div>
                    <div class="formation-value">{results["team1_formation"]}</div>
                    {runner_up_markup("team1")}
                </div>
                """,
                unsafe_allow_html=True,
            )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Tracked Video", "Team Clustering", "Phase Analysis", "Export Data"]
        )

        with tab1:
            st.markdown("### Player Tracking Output")
            st.caption("Annotated video showing detected players and tracking IDs.")
            st.video(str(results["tracked_video"]))
            st.caption(
                f"Formation analysis used the most stable tracked window: "
                f"frames {results['stable_window'][0]}–{results['stable_window'][1]}."
            )
        with tab2:
            st.markdown("### Team Clustering Output")
            st.caption("Average tracked player positions clustered into two teams.")
            st.image(str(results["team_plot"]), use_container_width=True)

        with tab3:
            st.markdown("### Attacking and Defending Shapes")
            st.caption(
                f"Ball detections: {results['ball_detections']} | "
                f"usable possession observations: {results['possession_observations']} | "
                f"smoothed possession frames: {results['smoothed_possession_observations']}"
            )

            phase_col_a, phase_col_b = st.columns(2)
            for team_id, phase_col in [(0, phase_col_a), (1, phase_col_b)]:
                phase = results["phase_results"][team_id]
                with phase_col:
                    st.markdown(
                        f"#### {results['team_labels'].get(team_id, f'Team {team_id}')}"
                    )

                    attack_summary = phase["attack_summary"]
                    defend_summary = phase["defend_summary"]

                    st.markdown("**Overall attacking shape**")
                    if attack_summary is not None:
                        st.success(
                            f"{attack_summary['shape']} "
                            f"({attack_summary['votes']}/{attack_summary['total_windows']} windows)"
                        )
                        st.caption(f"Vote distribution: {attack_summary['distribution']}")
                        if attack_summary["runner_ups"]:
                            runner_counts = pd.Series(
                                [item["formation"] for item in attack_summary["runner_ups"]]
                            ).value_counts()
                            st.caption(
                                f"Common secondary shape: {runner_counts.index[0]}"
                            )
                        if len(attack_summary["distribution"]) > 1:
                            with st.expander("Show other attacking shapes"):
                                for shape, votes in attack_summary["distribution"].items():
                                    st.write(f"{shape}: {votes} window(s)")
                    else:
                        st.info("Insufficient evidence for an overall attacking shape.")

                    st.markdown("**Overall defending shape**")
                    if defend_summary is not None:
                        st.success(
                            f"{defend_summary['shape']} "
                            f"({defend_summary['votes']}/{defend_summary['total_windows']} windows)"
                        )
                        st.caption(f"Vote distribution: {defend_summary['distribution']}")
                        if defend_summary["runner_ups"]:
                            runner_counts = pd.Series(
                                [item["formation"] for item in defend_summary["runner_ups"]]
                            ).value_counts()
                            st.caption(
                                f"Common secondary shape: {runner_counts.index[0]}"
                            )
                        if len(defend_summary["distribution"]) > 1:
                            with st.expander("Show other defending shapes"):
                                for shape, votes in defend_summary["distribution"].items():
                                    st.write(f"{shape}: {votes} window(s)")
                    else:
                        st.info("Insufficient evidence for an overall defending shape.")

                    st.markdown("**Representative windows**")
                    representative_attack = phase["attack_shape"]
                    representative_defend = phase["defend_shape"]
                    st.write(
                        {
                            "attacking_shape": representative_attack,
                            "attacking_window": phase["attack_window"],
                            "defending_shape": representative_defend,
                            "defending_window": phase["defend_window"],
                        }
                    )

        with tab4:
            st.markdown("### Download Results")
            st.caption("Download the generated team clustering data.")

            st.download_button(
                label="Download Team Cluster CSV",
                data=Path(results["team_csv"]).read_bytes(),
                file_name="team_clusters.csv",
                mime="text/csv",
            )

            st.download_button(
                label="Download Phase Summary JSON",
                data=Path(results["phase_summary_json"]).read_bytes(),
                file_name="phase_analysis_summary.json",
                mime="application/json",
            )

            st.download_button(
                label="Download Phase Window CSV",
                data=Path(results["phase_windows_csv"]).read_bytes(),
                file_name="phase_analysis_windows.csv",
                mime="text/csv",
            )


# Page routing

if st.session_state.page == "home":
    show_home()
else:
    show_dashboard()
