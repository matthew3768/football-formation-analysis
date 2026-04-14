import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/match_clip_5min.mkv")

def check_video(video_path: Path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else None

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Failed to read first frame.")

    return {
        "path": str(video_path),
        "resolution": f"{width}x{height}",
        "fps": fps,
        "frames": frame_count,
        "duration": duration,
    }

def main():
    info = check_video(VIDEO_PATH)
    print("Video info:")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("First frame read successfully ✅")

if __name__ == "__main__":
    main()