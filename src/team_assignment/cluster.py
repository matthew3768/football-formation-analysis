from pathlib import Path
import pandas as pd
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
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assign each average player position to one of two clusters.

    Args:
        player_positions: DataFrame with columns track_id, foot_x, foot_y
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with added 'team' column
    """
    required_cols = {"track_id", "foot_x", "foot_y"}
    if not required_cols.issubset(player_positions.columns):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    result = player_positions.copy()

    X = result[["foot_x", "foot_y"]]
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    result["team"] = kmeans.fit_predict(X)

    return result


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
    required_cols = {"track_id", "foot_x", "foot_y", "team"}
    if not required_cols.issubset(clustered_df.columns):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    image_out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        clustered_df["foot_x"],
        clustered_df["foot_y"],
        c=clustered_df["team"],
        s=100,
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