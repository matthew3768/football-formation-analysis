import pandas as pd

CSV_PATH = "outputs/tracking_clip_0_5.csv"
FPS = 25

df = pd.read_csv(CSV_PATH)

counts = (
    df.groupby("frame")["track_id"]
    .nunique()
    .rename("player_count")
    .reset_index()
)

# Fill missing frames
all_frames = pd.DataFrame({"frame": range(counts["frame"].min(), counts["frame"].max() + 1)})
counts = all_frames.merge(counts, on="frame", how="left").fillna(0)

def find_best_window(seconds):
    window = seconds * FPS

    tmp = counts.copy()
    tmp["rolling_mean"] = tmp["player_count"].rolling(window).mean()
    tmp["rolling_min"] = tmp["player_count"].rolling(window).min()

    # score prioritises stability
    tmp["score"] = tmp["rolling_mean"] + 0.5 * tmp["rolling_min"]

    best = tmp.loc[tmp["score"].idxmax()]

    end_frame = int(best["frame"])
    start_frame = end_frame - window

    print(f"\n=== BEST {seconds}s WINDOW ===")
    print(f"Start: {start_frame/FPS:.2f}s")
    print(f"End:   {end_frame/FPS:.2f}s")
    print(f"Mean players: {best['rolling_mean']:.2f}")
    print(f"Min players:  {int(best['rolling_min'])}")

for s in [10, 20, 30]:
    find_best_window(s)