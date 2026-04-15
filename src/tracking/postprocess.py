from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def clean_tracking_data(
    csv_in: Path,
    csv_out: Path,
    min_conf: float = 0.25,
    min_track_length: int = 5,
) -> Path:
    """
    Clean raw tracking CSV by:
    1. Removing weak detections
    2. Removing short-lived tracks

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

    # Remove weak detections
    df = df[df["conf"] >= min_conf].copy()
    print(f"Rows after confidence filter ({min_conf}): {len(df)}")

    # Remove short-lived tracks
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
    Create a quick sanity-check scatter plot of player foot positions
    for one frame.

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
    Print quick descriptive stats for number of tracked players per frame.
    """
    if not csv_in.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_in}")

    df = pd.read_csv(csv_in)
    counts = df.groupby("frame")["track_id"].nunique()

    print("\nPlayers per frame summary:")
    print(counts.describe())