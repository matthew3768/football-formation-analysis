from pathlib import Path
import csv
import cv2
from ultralytics import YOLO


class PlayerTracker:
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.5,
        imgsz: int = 960,
        tracker_config: str = "bytetrack.yaml",
        frame_skip: int = 1,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.tracker_config = tracker_config
        self.frame_skip = frame_skip

    def run(self, video_in: Path, video_out: Path, csv_out: Path) -> tuple[Path, Path]:
        if not video_in.exists():
            raise FileNotFoundError(f"Input video not found: {video_in}")

        video_out.parent.mkdir(parents=True, exist_ok=True)
        csv_out.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_in))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_in}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

        frame_idx = 0

        with open(csv_out, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "foot_x",
                "foot_y",
                "conf",
            ])

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                results = self.model.track(
                    source=frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.imgsz,
                    tracker=self.tracker_config,
                    persist=True,
                    verbose=False,
                    classes=[0],
                )[0]

                if results.boxes is not None and results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    ids = results.boxes.id.cpu().numpy().astype(int)
                    confs = results.boxes.conf.cpu().numpy()

                    for (x1, y1, x2, y2), track_id, conf in zip(boxes, ids, confs):
                        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

                        foot_x = (x1 + x2) / 2.0
                        foot_y = y2

                        writer.writerow([
                            frame_idx,
                            track_id,
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                            round(foot_x, 2),
                            round(foot_y, 2),
                            round(float(conf), 4),
                        ])

                        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
                        ifoot_x, ifoot_y = int(foot_x), int(foot_y)

                        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                        cv2.circle(frame, (ifoot_x, ifoot_y), 4, (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            f"ID {track_id} {conf:.2f}",
                            (ix1, max(20, iy1 - 5)),
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

        print(f"Tracking video saved to: {video_out}")
        print(f"Tracking CSV saved to: {csv_out}")
        return video_out, csv_out