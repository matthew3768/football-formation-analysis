from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def remove_team_outliers(team_df: pd.DataFrame, max_dist: float = 250) -> pd.DataFrame:
    team_df = team_df.copy()

    cx = team_df["foot_x"].median()
    cy = team_df["foot_y"].median()

    dist = np.sqrt(
        (team_df["foot_x"] - cx) ** 2 +
        (team_df["foot_y"] - cy) ** 2
    )

    team_df["dist"] = dist
    filtered = team_df[team_df["dist"] <= max_dist].copy()

    return filtered.drop(columns=["dist"])


def cluster_lines(team_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    team_df = team_df.copy()

    X = team_df[["foot_y"]]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    team_df["line"] = kmeans.fit_predict(X)

    return team_df


def cluster_lines_with_plausible_counts(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign outfield players to the most plausible common formation template.

    This is more stable than unconstrained KMeans on broadcast-frame y values,
    which can easily return shapes like 6-3-1 when perspective compresses one
    line of players.
    """
    templates = [
        (4, 4, 2),
        (4, 3, 3),
        (3, 5, 2),
        (4, 2, 3),  # useful when one nominal midfielder is very advanced
        (3, 4, 3),
        (5, 3, 2),
    ]

    ordered = team_df.sort_values("foot_y").copy()
    best_score = None
    best_assignment = None

    for template in templates:
        if sum(template) != len(ordered):
            continue

        start = 0
        score = 0.0
        assignments = []
        for line_idx, count in enumerate(template):
            group = ordered.iloc[start:start + count]
            score += float(((group["foot_y"] - group["foot_y"].mean()) ** 2).sum())
            assignments.extend([line_idx] * count)
            start += count

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignments

    if best_assignment is None:
        return cluster_lines(team_df, k=3)

    ordered["line"] = best_assignment
    return ordered


def sort_lines(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.copy()

    line_order = (
        team_df.groupby("line")["foot_y"]
        .mean()
        .sort_values()
        .index
    )

    mapping = {old: new for new, old in enumerate(line_order)}
    team_df["line"] = team_df["line"].map(mapping)

    return team_df


def reverse_lines(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.copy()

    max_line = team_df["line"].max()
    team_df["line"] = max_line - team_df["line"]

    return team_df


def normalise_formation_orientation(formation: str) -> str:
    """
    Prefer the conventional defensive-to-attacking reading of a shape.

    Camera orientation can flip one team relative to the other; if the reverse
    order starts with fewer defenders than it ends with, flip it back.
    """
    counts = [int(value) for value in formation.split("-")]
    if counts and counts[0] < counts[-1]:
        counts = list(reversed(counts))
    return "-".join(str(value) for value in counts)


def split_goalkeeper(team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_df = team_df.copy()

    gk_idx = team_df["foot_y"].idxmax()
    goalkeeper = team_df.loc[[gk_idx]].copy()
    outfield = team_df.drop(index=gk_idx).copy()

    return goalkeeper, outfield


def keep_best_outfield_tracks(team_df: pd.DataFrame, target_players: int = 10) -> pd.DataFrame:
    """
    Keep a stable outfield set for formation detection.

    The tracker can produce duplicate / drifting IDs. For formation work, it is
    better to keep the players closest to the team's central body than to let a
    few far-away ghost tracks distort the shape.
    """
    if len(team_df) <= target_players:
        return team_df.copy()

    team_df = team_df.copy()
    cx = team_df["foot_x"].median()
    cy = team_df["foot_y"].median()
    team_df["dist_to_team_centre"] = np.sqrt(
        (team_df["foot_x"] - cx) ** 2 +
        (team_df["foot_y"] - cy) ** 2
    )

    return (
        team_df.nsmallest(target_players, "dist_to_team_centre")
        .drop(columns=["dist_to_team_centre"])
        .copy()
    )


def format_formation(team_lines: pd.DataFrame) -> str:
    counts = team_lines.groupby("line").size().sort_index().tolist()
    return "-".join(str(x) for x in counts)


def detect_team_formation(
    team_df: pd.DataFrame,
    k: int = 3,
    max_dist: float = 250,
    reverse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Detect formation lines for one team.

    Returns:
        goalkeeper_df, outfield_lines_df, formation_string
    """
    goalkeeper, outfield = split_goalkeeper(team_df)
    outfield = keep_best_outfield_tracks(outfield, target_players=10)

    # Only apply distance pruning if we still have more than a normal outfield
    # unit. This avoids deleting genuine wide players from a valid XI.
    if len(outfield) > 10:
        outfield = remove_team_outliers(outfield, max_dist=max_dist)

    if len(outfield) < k:
        raise ValueError(
            f"Not enough outfield players to detect {k} formation lines: "
            f"found {len(outfield)}"
        )

    if len(outfield) == 10:
        outfield = cluster_lines_with_plausible_counts(outfield)
    else:
        outfield = cluster_lines(outfield, k=k)
        outfield = sort_lines(outfield)

    if reverse:
        outfield = reverse_lines(outfield)

    goalkeeper = goalkeeper.copy()
    goalkeeper["line"] = -1

    formation = format_formation(outfield)
    formation = normalise_formation_orientation(formation)

    return goalkeeper, outfield, formation


def detect_formations(
    clustered_df: pd.DataFrame,
    k: int = 3,
    max_dist: float = 250,
    reverse_team_1: bool = True,
) -> dict:
    """
    Detect formations for both teams from clustered player positions.

    clustered_df must contain:
    - foot_x
    - foot_y
    - team
    """
    team0 = clustered_df[clustered_df["team"] == 0].copy()
    team1 = clustered_df[clustered_df["team"] == 1].copy()

    gk0, team0_lines, formation0 = detect_team_formation(
        team0, k=k, max_dist=max_dist, reverse=False
    )

    gk1, team1_lines, formation1 = detect_team_formation(
        team1, k=k, max_dist=max_dist, reverse=reverse_team_1
    )

    team0_final = pd.concat([team0_lines, gk0], ignore_index=True)
    team1_final = pd.concat([team1_lines, gk1], ignore_index=True)

    return {
        "team0_goalkeeper": gk0,
        "team1_goalkeeper": gk1,
        "team0_lines": team0_lines,
        "team1_lines": team1_lines,
        "team0_final": team0_final,
        "team1_final": team1_final,
        "team0_formation": formation0,
        "team1_formation": formation1,
    }
