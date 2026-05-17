from pathlib import Path
import csv
import cv2
import numpy as np
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
        filter_to_pitch: bool = True,
        pitch_erode_px: int = 5,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.tracker_config = tracker_config
        self.frame_skip = frame_skip
        self.filter_to_pitch = filter_to_pitch
        self.pitch_erode_px = pitch_erode_px

    def _build_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Build a binary mask for the visible grass area.

        This keeps the implementation lightweight and camera-agnostic:
        players are retained only when their foot point lands on the pitch.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # A broad green range is more robust to shade and sunlight than a tight threshold.
        lower_green = np.array([25, 35, 35])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitch_mask = np.zeros_like(mask)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(pitch_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            if self.pitch_erode_px > 0:
                erode_kernel = np.ones(
                    (self.pitch_erode_px, self.pitch_erode_px),
                    np.uint8,
                )
                pitch_mask = cv2.erode(pitch_mask, erode_kernel, iterations=1)

        return pitch_mask

    def run(
        self,
        video_in: Path,
        video_out: Path,
        csv_out: Path,
        progress_callback=None,
    ) -> tuple[Path, Path]:
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

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

                pitch_mask = self._build_pitch_mask(frame) if self.filter_to_pitch else None

                if results.boxes is not None and results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    ids = results.boxes.id.cpu().numpy().astype(int)
                    confs = results.boxes.conf.cpu().numpy()

                    for (x1, y1, x2, y2), track_id, conf in zip(boxes, ids, confs):
                        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

                        foot_x = (x1 + x2) / 2.0
                        foot_y = y2

                        if pitch_mask is not None:
                            mask_x = int(np.clip(round(foot_x), 0, width - 1))
                            mask_y = int(np.clip(round(foot_y), 0, height - 1))
                            if pitch_mask[mask_y, mask_x] == 0:
                                continue

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
                if progress_callback is not None and total_frames > 0:
                    progress_callback(min(frame_idx / total_frames, 1.0), frame_idx, total_frames)
                if frame_idx % 100 == 0:
                    print(f"Tracked {frame_idx} frames...")

        cap.release()
        out.release()

        print(f"Tracking video saved to: {video_out}")
        print(f"Tracking CSV saved to: {csv_out}")
        return video_out, csv_out
