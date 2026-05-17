from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def detect_ball_positions(
    video_in: Path,
    conf_thresh: float = 0.10,
    frame_skip: int = 1,
    model_path: str = "yolov8s.pt",
) -> pd.DataFrame:
    """
    Detect the football through the clip.

    COCO class 32 = sports ball. Broadcast footage is difficult, so this
    returns sparse detections rather than pretending every frame is solvable.
    """
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for ball detection: {video_in}")

    model = YOLO(model_path)
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        result = model.predict(
            frame,
            conf=conf_thresh,
            classes=[32],
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[best_idx]
            rows.append(
                {
                    "frame": frame_idx,
                    "ball_x": float((x1 + x2) / 2.0),
                    "ball_y": float((y1 + y2) / 2.0),
                    "ball_conf": float(confs[best_idx]),
                }
            )

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def estimate_possession(
    tracking_csv: Path,
    team_assignments: pd.DataFrame,
    ball_positions: pd.DataFrame,
    max_player_distance: float = 180.0,
) -> pd.DataFrame:
    """
    Assign sparse ball detections to the nearest known tracked team.
    """
    tracking_df = pd.read_csv(tracking_csv)
    team_map = team_assignments.set_index("track_id")["team"].to_dict()
    tracking_df = tracking_df[tracking_df["track_id"].isin(team_map)].copy()
    tracking_df["team"] = tracking_df["track_id"].map(team_map)

    rows = []
    for _, ball in ball_positions.iterrows():
        frame_rows = tracking_df[tracking_df["frame"] == ball["frame"]].copy()
        if frame_rows.empty:
            continue

        frame_rows["dist_to_ball"] = np.sqrt(
            (frame_rows["foot_x"] - ball["ball_x"]) ** 2
            + (frame_rows["foot_y"] - ball["ball_y"]) ** 2
        )
        nearest = frame_rows.nsmallest(1, "dist_to_ball").iloc[0]
        if float(nearest["dist_to_ball"]) > max_player_distance:
            continue

        rows.append(
            {
                "frame": int(ball["frame"]),
                "possession_team": int(nearest["team"]),
                "dist_to_ball": float(nearest["dist_to_ball"]),
                "ball_conf": float(ball["ball_conf"]),
            }
        )

    return pd.DataFrame(rows)


def smooth_possession(
    possession_df: pd.DataFrame,
    max_gap_frames: int = 20,
    min_run_length: int = 2,
) -> pd.DataFrame:
    """
    Fill short gaps between same-team possession observations.

    This treats possession as a temporal state rather than a frame-by-frame
    coincidence, while remaining conservative around actual turnovers.
    """
    if possession_df.empty:
        return possession_df.copy()

    df = possession_df.sort_values("frame").copy()
    rows = df.to_dict("records")
    smoothed_rows = []

    for idx, row in enumerate(rows):
        smoothed_rows.append(row)

        if idx == len(rows) - 1:
            continue

        next_row = rows[idx + 1]
        same_team = row["possession_team"] == next_row["possession_team"]
        gap = int(next_row["frame"] - row["frame"])

        if same_team and 1 < gap <= max_gap_frames:
            for frame in range(int(row["frame"]) + 1, int(next_row["frame"])):
                smoothed_rows.append(
                    {
                        "frame": frame,
                        "possession_team": int(row["possession_team"]),
                        "dist_to_ball": np.nan,
                        "ball_conf": np.nan,
                    }
                )

    smoothed = pd.DataFrame(smoothed_rows).sort_values("frame").reset_index(drop=True)

    # Short isolated runs are more likely detector noise than real possession.
    run_id = (smoothed["possession_team"] != smoothed["possession_team"].shift()).cumsum()
    run_lengths = smoothed.groupby(run_id)["frame"].transform("count")
    smoothed = smoothed[run_lengths >= min_run_length].copy()

    return smoothed.reset_index(drop=True)


def select_phase_window(
    tracking_csv: Path,
    possession_df: pd.DataFrame,
    target_team: int,
    analysis_team: int | None = None,
    team_assignments: pd.DataFrame | None = None,
    window_size: int = 250,
    step_size: int = 25,
    min_tracks: int = 8,
    min_analysis_team_tracks: int = 10,
    min_opponent_tracks: int = 6,
    min_possession_observations: int = 3,
    min_target_share: float = 0.60,
) -> tuple[int, int] | None:
    """
    Select a stable window where one side is usually in possession.
    """
    tracking_df = pd.read_csv(tracking_csv)
    if tracking_df.empty or possession_df.empty:
        return None

    if team_assignments is not None:
        team_map = team_assignments.set_index("track_id")["team"].to_dict()
        tracking_df = tracking_df[tracking_df["track_id"].isin(team_map)].copy()
        tracking_df["team"] = tracking_df["track_id"].map(team_map)

    min_frame = int(tracking_df["frame"].min())
    max_frame = int(tracking_df["frame"].max())

    best_window = None
    best_score = None

    for start in range(min_frame, max_frame - window_size + 2, step_size):
        end = start + window_size - 1
        window_tracks = tracking_df[
            (tracking_df["frame"] >= start) & (tracking_df["frame"] <= end)
        ].copy()
        window_possession = possession_df[
            (possession_df["frame"] >= start) & (possession_df["frame"] <= end)
        ].copy()

        if len(window_possession) < min_possession_observations:
            continue

        target_share = float((window_possession["possession_team"] == target_team).mean())
        if target_share < min_target_share:
            continue

        per_frame_counts = window_tracks.groupby("frame")["track_id"].nunique()
        if per_frame_counts.median() < min_tracks:
            continue

        if team_assignments is not None:
            visible_team_counts = (
                window_tracks.groupby("team")["track_id"]
                .nunique()
                .to_dict()
            )
            team_to_analyse = target_team if analysis_team is None else analysis_team
            opponent_team = 1 - team_to_analyse
            if visible_team_counts.get(team_to_analyse, 0) < min_analysis_team_tracks:
                continue
            if visible_team_counts.get(opponent_team, 0) < min_opponent_tracks:
                continue

        per_track_spread = (
            window_tracks.groupby("track_id")[["foot_x", "foot_y"]]
            .std()
            .fillna(0.0)
        )
        mean_spread = float(
            np.sqrt(per_track_spread["foot_x"] ** 2 + per_track_spread["foot_y"] ** 2).mean()
        )
        count_instability = float(per_frame_counts.std(ddof=0) or 0.0)

        score = mean_spread + (count_instability * 10.0) - (target_share * 25.0)
        if best_score is None or score < best_score:
            best_score = score
            best_window = (start, end)

    return best_window


def collect_phase_windows(
    tracking_csv: Path,
    possession_df: pd.DataFrame,
    target_team: int,
    analysis_team: int | None = None,
    team_assignments: pd.DataFrame | None = None,
    window_size: int = 250,
    step_size: int = 25,
    min_tracks: int = 8,
    min_analysis_team_tracks: int = 10,
    min_opponent_tracks: int = 6,
    min_possession_observations: int = 3,
    min_target_share: float = 0.60,
) -> list[dict]:
    """
    Return every valid phase window, ranked from strongest to weakest.
    """
    tracking_df = pd.read_csv(tracking_csv)
    if tracking_df.empty or possession_df.empty:
        return []

    if team_assignments is not None:
        team_map = team_assignments.set_index("track_id")["team"].to_dict()
        tracking_df = tracking_df[tracking_df["track_id"].isin(team_map)].copy()
        tracking_df["team"] = tracking_df["track_id"].map(team_map)

    min_frame = int(tracking_df["frame"].min())
    max_frame = int(tracking_df["frame"].max())
    windows = []

    for start in range(min_frame, max_frame - window_size + 2, step_size):
        end = start + window_size - 1
        window_tracks = tracking_df[
            (tracking_df["frame"] >= start) & (tracking_df["frame"] <= end)
        ].copy()
        window_possession = possession_df[
            (possession_df["frame"] >= start) & (possession_df["frame"] <= end)
        ].copy()

        if len(window_possession) < min_possession_observations:
            continue

        target_share = float((window_possession["possession_team"] == target_team).mean())
        if target_share < min_target_share:
            continue

        per_frame_counts = window_tracks.groupby("frame")["track_id"].nunique()
        if per_frame_counts.median() < min_tracks:
            continue

        visible_team_counts = {}
        if team_assignments is not None:
            visible_team_counts = (
                window_tracks.groupby("team")["track_id"]
                .nunique()
                .to_dict()
            )
            team_to_analyse = target_team if analysis_team is None else analysis_team
            opponent_team = 1 - team_to_analyse
            if visible_team_counts.get(team_to_analyse, 0) < min_analysis_team_tracks:
                continue
            if visible_team_counts.get(opponent_team, 0) < min_opponent_tracks:
                continue

        per_track_spread = (
            window_tracks.groupby("track_id")[["foot_x", "foot_y"]]
            .std()
            .fillna(0.0)
        )
        mean_spread = float(
            np.sqrt(per_track_spread["foot_x"] ** 2 + per_track_spread["foot_y"] ** 2).mean()
        )
        count_instability = float(per_frame_counts.std(ddof=0) or 0.0)
        score = mean_spread + (count_instability * 10.0) - (target_share * 25.0)

        windows.append(
            {
                "window": (start, end),
                "score": score,
                "target_share": target_share,
                "visible_team_counts": visible_team_counts,
            }
        )

    return sorted(windows, key=lambda item: item["score"])


def average_positions_for_tracks(
    tracking_csv: Path,
    team_assignments: pd.DataFrame,
    frame_range: tuple[int, int],
) -> pd.DataFrame:
    """
    Recompute average positions for known players inside a phase window.
    """
    start_frame, end_frame = frame_range
    tracking_df = pd.read_csv(tracking_csv)
    tracking_df = tracking_df[
        (tracking_df["frame"] >= start_frame)
        & (tracking_df["frame"] <= end_frame)
    ].copy()

    track_ids = team_assignments["track_id"].tolist()
    tracking_df = tracking_df[tracking_df["track_id"].isin(track_ids)].copy()

    averaged = (
        tracking_df.groupby("track_id")[["foot_x", "foot_y"]]
        .mean()
        .reset_index()
    )

    metadata_cols = ["track_id", "team", "team_colour"]
    return averaged.merge(team_assignments[metadata_cols], on="track_id", how="inner")
