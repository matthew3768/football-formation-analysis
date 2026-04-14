from pathlib import Path
import cv2
from ultralytics import YOLO


class PlayerDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def run(self, video_in: Path, video_out: Path) -> Path:
        """
        Run player detection on a video and save an annotated output video.
        Only COCO class 0 ('person') detections are drawn.

        Args:
            video_in: Path to input video
            video_out: Path to output annotated video

        Returns:
            Path to saved output video
        """
        if not video_in.exists():
            raise FileNotFoundError(f"Input video not found: {video_in}")

        video_out.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_in))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_in}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = self.model.predict(
                frame,
                conf=self.conf_thresh,
                verbose=False
            )[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 0 = person
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {conf:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            out.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        out.release()

        print(f"Detection output saved to: {video_out}")
        return video_out