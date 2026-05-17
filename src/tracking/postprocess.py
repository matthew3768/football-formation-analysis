from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_tracking_data(
    csv_in: Path,
    csv_out: Path,
    min_conf: float = 0.25,
    min_track_length: int = 5,
) -> Path:
    """
    Filter unreliable detections and tracks from the raw tracking output.

    Args:
        csv_in: Path to raw tracking CSV
        csv_out: Path to cleaned tracking CSV
        min_conf: Minimum confidence to keep a detection
        min_track_length: Minimum number of frames a track must appear in

    Returns:
        Path to cleaned CSV
    """
    if not csv_in.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_in}")

    csv_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)

    print(f"Raw rows: {len(df)}")

    df = df[df["conf"] >= min_conf].copy()
    print(f"Rows after confidence filter ({min_conf}): {len(df)}")

    track_counts = df["track_id"].value_counts()
    valid_ids = track_counts[track_counts >= min_track_length].index
    df = df[df["track_id"].isin(valid_ids)].copy()
    print(f"Rows after track length filter ({min_track_length}): {len(df)}")
    print(f"Remaining unique tracks: {df['track_id'].nunique()}")

    df.to_csv(csv_out, index=False)
    print(f"Cleaned tracking CSV saved to: {csv_out}")

    return csv_out


def plot_frame_positions(
    csv_in: Path,
    frame_number: int,
    image_out: Path,
) -> Path:
    """
    Plot tracked player foot positions for one frame as a quick visual check.

    Args:
        csv_in: Path to cleaned tracking CSV
        frame_number: Frame number to plot
        image_out: Output image path

    Returns:
        Path to saved plot image
    """
    if not csv_in.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_in}")

    image_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)
    frame_df = df[df["frame"] == frame_number].copy()

    if frame_df.empty:
        raise ValueError(f"No detections found for frame {frame_number}")

    plt.figure(figsize=(10, 6))
    plt.scatter(frame_df["foot_x"], frame_df["foot_y"])
    plt.gca().invert_yaxis()

    for _, row in frame_df.iterrows():
        plt.text(row["foot_x"], row["foot_y"], str(int(row["track_id"])), fontsize=8)

    plt.title(f"Tracked player positions - frame {frame_number}")
    plt.xlabel("foot_x")
    plt.ylabel("foot_y")
    plt.tight_layout()
    plt.savefig(image_out, dpi=200)
    plt.close()

    print(f"Sanity-check plot saved to: {image_out}")
    return image_out


def summarise_players_per_frame(csv_in: Path) -> None:
    """
    Print descriptive statistics for tracked-player counts per frame.
    """
    if not csv_in.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_in}")

    df = pd.read_csv(csv_in)
    counts = df.groupby("frame")["track_id"].nunique()

    print("\nPlayers per frame summary:")
    print(counts.describe())


def select_stable_window(
    csv_in: Path,
    window_size: int = 250,
    step_size: int = 25,
    min_tracks: int = 18,
) -> tuple[int, int]:
    """
    Find the calmest contiguous window of play for formation analysis.

    Windows are rewarded for:
    - keeping many tracked players visible
    - maintaining similar player spacing from frame to frame

    Lower score is better.
    """
    if not csv_in.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_in}")

    df = pd.read_csv(csv_in)
    if df.empty:
        raise ValueError("Tracking CSV is empty")

    min_frame = int(df["frame"].min())
    max_frame = int(df["frame"].max())

    best_window = None
    best_score = None

    for start in range(min_frame, max_frame - window_size + 2, step_size):
        end = start + window_size - 1
        window_df = df[(df["frame"] >= start) & (df["frame"] <= end)].copy()
        if window_df.empty:
            continue

        per_frame_counts = window_df.groupby("frame")["track_id"].nunique()
        median_tracks = float(per_frame_counts.median())
        if median_tracks < min_tracks:
            continue

        # Spread measures whether players remain in a tactically stable shape.
        per_track_spread = (
            window_df.groupby("track_id")[["foot_x", "foot_y"]]
            .std()
            .fillna(0.0)
        )
        mean_spread = float(
            np.sqrt(per_track_spread["foot_x"] ** 2 + per_track_spread["foot_y"] ** 2).mean()
        )

        # Count instability captures tracking flicker across the window.
        count_instability = float(per_frame_counts.std(ddof=0) or 0.0)

        score = mean_spread + (count_instability * 10.0)

        if best_score is None or score < best_score:
            best_score = score
            best_window = (start, end)

    if best_window is None:
        # Fall back to the full clip if no window satisfies the thresholds.
        return min_frame, max_frame

    return best_window
