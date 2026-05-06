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


def split_goalkeeper(team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_df = team_df.copy()

    gk_idx = team_df["foot_y"].idxmax()
    goalkeeper = team_df.loc[[gk_idx]].copy()
    outfield = team_df.drop(index=gk_idx).copy()

    return goalkeeper, outfield


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
    outfield = remove_team_outliers(outfield, max_dist=max_dist)
    outfield = cluster_lines(outfield, k=k)
    outfield = sort_lines(outfield)

    if reverse:
        outfield = reverse_lines(outfield)

    goalkeeper = goalkeeper.copy()
    goalkeeper["line"] = -1

    formation = format_formation(outfield)

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