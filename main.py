from pathlib import Path

from src.detection.detector import PlayerDetector


BASE_DIR = Path(__file__).resolve().parent
VIDEO_IN = BASE_DIR / "data" / "clips" / "clip_0_5.mp4"
VIDEO_OUT = BASE_DIR / "outputs" / "detect_clip_0_5.mp4"


def main() -> None:
    detector = PlayerDetector(
        model_path="yolov8n.pt",
        conf_thresh=0.35,
    )
    detector.run(VIDEO_IN, VIDEO_OUT)


if __name__ == "__main__":
    main()