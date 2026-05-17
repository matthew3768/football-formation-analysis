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


def cluster_lines(team_df: pd.DataFrame, k: int = 3, axis_col: str = "foot_x") -> pd.DataFrame:
    team_df = team_df.copy()

    X = team_df[[axis_col]]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    team_df["line"] = kmeans.fit_predict(X)

    return team_df


def cluster_lines_with_plausible_counts(
    team_df: pd.DataFrame,
    axis_col: str = "foot_x",
    ascending: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Assign outfield players to the most plausible common formation template.
    """
    templates = [
        (4, 4, 2),
        (4, 3, 3),
        (3, 5, 2),
        (3, 4, 3),
        (5, 3, 2),
        (4, 2, 3, 1),
        (4, 1, 4, 1),
    ]

    ordered = team_df.sort_values(axis_col, ascending=ascending).copy()
    best_score = None
    best_assignment = None
    best_template = None
    candidate_results = []

    for template in templates:
        if sum(template) != len(ordered):
            continue

        start = 0
        score = 0.0
        assignments = []
        for line_idx, count in enumerate(template):
            group = ordered.iloc[start:start + count]
            score += float(((group[axis_col] - group[axis_col].mean()) ** 2).sum())
            assignments.extend([line_idx] * count)
            start += count

        # Mild prior: four-back systems are more common, so a three/five-back
        # shape should only win when it fits meaningfully better.
        if template[0] == 4:
            score *= 0.95

        # More detailed shapes should earn their extra complexity. Without this,
        # 4-line templates nearly always win because they can partition the same
        # players more finely than 3-line templates.
        if len(template) == 4:
            score *= 1.20

        candidate_results.append((template, score, assignments))

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignments
            best_template = template

    # Require a 4-line shape to beat the best 3-line alternative by a clear
    # margin; otherwise prefer the simpler tactical reading.
    if best_template is not None and len(best_template) == 4:
        three_line_candidates = [item for item in candidate_results if len(item[0]) == 3]
        if three_line_candidates:
            best_three_template, best_three_score, best_three_assignment = min(
                three_line_candidates,
                key=lambda item: item[1],
            )
            if best_score > best_three_score * 0.80:
                best_template = best_three_template
                best_score = best_three_score
                best_assignment = best_three_assignment

    if best_assignment is None:
        return cluster_lines(team_df, k=3, axis_col=axis_col), []

    ordered["line"] = best_assignment
    ranked_candidates = sorted(candidate_results, key=lambda item: item[1])
    candidate_summary = [
        {
            "formation": "-".join(str(value) for value in template),
            "score": float(score),
        }
        for template, score, _ in ranked_candidates
    ]
    return ordered, candidate_summary


def sort_lines(
    team_df: pd.DataFrame,
    axis_col: str = "foot_x",
    ascending: bool = True,
) -> pd.DataFrame:
    team_df = team_df.copy()

    line_order = (
        team_df.groupby("line")[axis_col]
        .mean()
        .sort_values(ascending=ascending)
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


def split_goalkeeper(
    team_df: pd.DataFrame,
    defending_left: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_df = team_df.copy()

    gk_idx = team_df["foot_x"].idxmin() if defending_left else team_df["foot_x"].idxmax()
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
    defending_left: bool,
    k: int = 3,
    max_dist: float = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict]:
    """
    Detect formation lines for one team.

    Returns:
        goalkeeper_df, outfield_lines_df, formation_string
    """
    goalkeeper, outfield = split_goalkeeper(team_df, defending_left=defending_left)
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
        outfield, candidate_summary = cluster_lines_with_plausible_counts(
            outfield,
            axis_col="foot_x",
            ascending=defending_left,
        )
    else:
        outfield = cluster_lines(outfield, k=k, axis_col="foot_x")
        outfield = sort_lines(
            outfield,
            axis_col="foot_x",
            ascending=defending_left,
        )
        candidate_summary = []

    goalkeeper = goalkeeper.copy()
    goalkeeper["line"] = -1

    formation = format_formation(outfield)

    diagnostics = {
        "best": candidate_summary[0] if candidate_summary else None,
        "runner_up": candidate_summary[1] if len(candidate_summary) > 1 else None,
        "score_gap": (
            candidate_summary[1]["score"] - candidate_summary[0]["score"]
            if len(candidate_summary) > 1
            else None
        ),
        "candidates": candidate_summary,
    }

    return goalkeeper, outfield, formation, diagnostics


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

    team0_defending_left = team0["foot_x"].median() < team1["foot_x"].median()
    team1_defending_left = not team0_defending_left

    gk0, team0_lines, formation0, diagnostics0 = detect_team_formation(
        team0,
        defending_left=team0_defending_left,
        k=k,
        max_dist=max_dist,
    )

    gk1, team1_lines, formation1, diagnostics1 = detect_team_formation(
        team1,
        defending_left=team1_defending_left,
        k=k,
        max_dist=max_dist,
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
        "team0_diagnostics": diagnostics0,
        "team1_diagnostics": diagnostics1,
    }
