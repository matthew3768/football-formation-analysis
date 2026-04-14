from pathlib import Path
import cv2
from ultralytics import YOLO


class PlayerTracker:
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        imgsz: int = 1280,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.tracker_config = tracker_config

    def run(self, video_in: Path, video_out: Path) -> Path:
        """
        Run player tracking on a video and save an annotated output video.

        Args:
            video_in: Input video path
            video_out: Output annotated video path

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

            results = self.model.track(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.imgsz,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
                classes=[0],  # person only
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID {track_id} {conf:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            out.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Tracked {frame_idx} frames...")

        cap.release()
        out.release()

        print(f"Tracking output saved to: {video_out}")
        return video_out