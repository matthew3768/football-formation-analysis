from pathlib import Path

from src.detection.detector import PlayerDetector
from src.tracking.postprocess import (
    clean_tracking_data,
    plot_frame_positions,
    summarise_players_per_frame,
)


BASE_DIR = Path(__file__).resolve().parent
VIDEO_IN = BASE_DIR / "data" / "clips" / "best_segment.mp4"
#VIDEO_OUT = BASE_DIR / "outputs" / "detect_clip_0_5.mp4"
VIDEO_OUT = BASE_DIR / "outputs" / "best_segment_tracking_clip_0_1.mp4"
RAW_CSV_OUT = BASE_DIR / "outputs" / "best_segment_tracking_clip_0_1.csv"
CLEAN_CSV_OUT = BASE_DIR / "outputs" / "best_segment_tracking_clip_0_1_clean.csv"
PLOT_OUT = BASE_DIR / "outputs" / "frame_500_positions.png"


# def main() -> None:
#     detector = PlayerDetector(
#         model_path="yolov8n.pt",
#         conf_thresh=0.25,
#     )
#     detector.run(VIDEO_IN, VIDEO_OUT)

def main() -> None:
    from src.tracking.tracker import PlayerTracker

    tracker = PlayerTracker(
        model_path="yolov8s.pt",
        conf_thresh=0.25,
        iou_thresh=0.5,
        #imgsz=1280,
        imgsz=960,
        tracker_config="bytetrack.yaml",
        frame_skip=1,
    )
    tracker.run(VIDEO_IN, VIDEO_OUT, RAW_CSV_OUT)

    clean_tracking_data(
        csv_in=RAW_CSV_OUT,
        csv_out=CLEAN_CSV_OUT,
        min_conf=0.25,
        min_track_length=5,
    )

    summarise_players_per_frame(CLEAN_CSV_OUT)

    plot_frame_positions(
        csv_in=CLEAN_CSV_OUT,
        frame_number=500,
        image_out=PLOT_OUT,
    )

if __name__ == "__main__":
    main()