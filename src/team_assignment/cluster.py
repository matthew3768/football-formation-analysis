from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def compute_average_positions(
    csv_in,
    csv_out=None,
    min_samples_per_track=20,
    max_tracks=24,
):
    df = pd.read_csv(csv_in)

    track_counts = df["track_id"].value_counts()
    valid_ids = track_counts[track_counts >= min_samples_per_track].index
    df = df[df["track_id"].isin(valid_ids)].copy()

    # Keep only the longest-lasting tracks
    top_ids = df["track_id"].value_counts().head(max_tracks).index
    df = df[df["track_id"].isin(top_ids)].copy()

    player_positions = (
        df.groupby("track_id")[["foot_x", "foot_y"]]
        .mean()
        .reset_index()
    )

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        player_positions.to_csv(csv_out, index=False)

    return player_positions


def cluster_teams(
    player_positions: pd.DataFrame,
    tracking_csv: Path,
    video_in: Path,
    random_state: int = 42,
    max_samples_per_track: int = 20,
) -> pd.DataFrame:
    """
    Assign each tracked player to one of two teams using kit colour.

    Args:
        player_positions: DataFrame with columns track_id, foot_x, foot_y
        tracking_csv: Tracking CSV containing bounding boxes for each track
        video_in: Source match clip used to extract jersey crops
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with added colour features plus 'team' column
    """
    required_cols = {"track_id", "foot_x", "foot_y"}
    if not required_cols.issubset(player_positions.columns):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    result = player_positions.copy()
    colour_features = extract_track_jersey_colours(
        tracking_csv=tracking_csv,
        video_in=video_in,
        track_ids=result["track_id"].tolist(),
        max_samples_per_track=max_samples_per_track,
    )

    result = result.merge(colour_features, on="track_id", how="left")

    if result[["lab_l", "lab_a", "lab_b"]].isna().any().any():
        missing = result[result["lab_l"].isna()]["track_id"].tolist()
        raise ValueError(f"Could not extract jersey colours for tracks: {missing}")

    if len(result) < 2:
        raise ValueError("Not enough valid jersey-colour tracks to cluster teams")

    X = result[["lab_l", "lab_a", "lab_b"]]
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    result["team"] = kmeans.fit_predict(X)

    team_colours = (
        result.groupby("team")[["rgb_r", "rgb_g", "rgb_b"]]
        .median()
        .round()
        .astype(int)
    )
    result["team_colour"] = result["team"].map(
        lambda team: _bright_hex_colour(team_colours.loc[team].tolist())
    )

    return result


def extract_track_jersey_colours(
    tracking_csv: Path,
    video_in: Path,
    track_ids: list[int],
    max_samples_per_track: int = 20,
) -> pd.DataFrame:
    """
    Estimate one representative jersey colour per track.

    We sample the upper-middle body region from several frames, remove obvious
    green grass pixels, then use the median colour to reduce lighting noise.
    """
    tracking_df = pd.read_csv(tracking_csv)
    tracking_df = tracking_df[tracking_df["track_id"].isin(track_ids)].copy()

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for colour extraction: {video_in}")

    samples_by_track: dict[int, list[np.ndarray]] = {int(track_id): [] for track_id in track_ids}

    sampled_track_rows = []
    for track_id in track_ids:
        track_rows = tracking_df[tracking_df["track_id"] == track_id]
        if track_rows.empty:
            continue

        sampled_rows = track_rows.iloc[
            np.linspace(
                0,
                len(track_rows) - 1,
                num=min(max_samples_per_track, len(track_rows)),
                dtype=int,
            )
        ]
        sampled_track_rows.append(sampled_rows)

    if not sampled_track_rows:
        cap.release()
        return pd.DataFrame()

    sampled_df = pd.concat(sampled_track_rows, ignore_index=True).sort_values("frame")
    rows_by_frame = {
        int(frame): frame_rows.copy()
        for frame, frame_rows in sampled_df.groupby("frame")
    }

    target_frames = set(rows_by_frame.keys())
    frame_idx = 0

    while target_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx not in target_frames:
            frame_idx += 1
            continue

        for _, row in rows_by_frame[frame_idx].iterrows():
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # Torso crop: avoids boots, shorts, and a lot of grass leakage.
            crop_x1 = x1 + int(box_w * 0.2)
            crop_x2 = x2 - int(box_w * 0.2)
            crop_y1 = y1 + int(box_h * 0.15)
            crop_y2 = y1 + int(box_h * 0.55)
            crop = frame[max(0, crop_y1):max(0, crop_y2), max(0, crop_x1):max(0, crop_x2)]

            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            non_green_mask = cv2.bitwise_not(
                cv2.inRange(hsv, np.array([25, 35, 35]), np.array([95, 255, 255]))
            )

            pixels_bgr = crop[non_green_mask > 0]
            if len(pixels_bgr) < 10:
                pixels_bgr = crop.reshape(-1, 3)

            median_bgr = np.median(pixels_bgr, axis=0).astype(np.uint8)
            samples_by_track[int(track_id)].append(median_bgr)

        target_frames.remove(frame_idx)
        frame_idx += 1

    cap.release()

    rows = []
    for track_id, samples in samples_by_track.items():
        if not samples:
            continue

        median_bgr = np.median(np.array(samples), axis=0).astype(np.uint8)
        lab = cv2.cvtColor(median_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        b, g, r = median_bgr.tolist()

        rows.append(
            {
                "track_id": track_id,
                "rgb_r": int(r),
                "rgb_g": int(g),
                "rgb_b": int(b),
                "lab_l": int(lab[0]),
                "lab_a": int(lab[1]),
                "lab_b": int(lab[2]),
            }
        )

    return pd.DataFrame(rows)
def _bright_hex_colour(rgb_values: list[int]) -> str:
    """
    Make learned team colours readable on a plot while preserving hue.
    """
    rgb = np.uint8([[rgb_values]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[0, 0, 1] = max(int(hsv[0, 0, 1]), 170)
    hsv[0, 0, 2] = max(int(hsv[0, 0, 2]), 190)
    bright_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return "#{:02x}{:02x}{:02x}".format(*bright_rgb.tolist())


def save_team_assignments(
    clustered_df: pd.DataFrame,
    csv_out: Path,
) -> Path:
    """
    Save clustered player positions to CSV.
    """
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(csv_out, index=False)
    return csv_out


def plot_team_clusters(
    clustered_df: pd.DataFrame,
    image_out: Path,
    title: str = "Team clustering from average player positions",
) -> Path:
    """
    Plot clustered player positions.
    """
    required_cols = {"track_id", "foot_x", "foot_y", "team", "team_colour"}
    if not required_cols.issubset(clustered_df.columns):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    image_out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        clustered_df["foot_x"],
        clustered_df["foot_y"],
        c=clustered_df["team_colour"],
        s=100,
        edgecolors="black",
        linewidths=0.6,
    )
    plt.gca().invert_yaxis()

    for _, row in clustered_df.iterrows():
        plt.text(
            row["foot_x"],
            row["foot_y"],
            str(int(row["track_id"])),
            fontsize=8,
        )

    plt.title(title)
    plt.xlabel("foot_x")
    plt.ylabel("foot_y")
    plt.tight_layout()
    plt.savefig(image_out, dpi=200)
    plt.close()

    return image_out
