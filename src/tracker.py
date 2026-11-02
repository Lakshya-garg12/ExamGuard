"""
ExamGuard - Multi-Object Temporal Tracking Engine
Maintains persistent IDs across video frames, tracks object dwell time,
implements temporal smoothing (prevents box flicker), and coordinates with AlertManager.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.config import CLASS_CONFIG, MIN_DETECTION_FRAMES
from src.alerts import alert_manager
from src.detector import compute_iou

class ObjectTracker:
    def __init__(self, detector):
        self.detector = detector
        self.next_track_id = 1
        # Active tracks: {track_id: {"bbox": ..., "class_name": ..., "frames_seen": ..., "lost_frames": ...}}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.max_lost_frames = 12  # Keep track alive for up to 12 lost frames to stop flickering

    def track_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "ExamHall-Cam-01",
        conf: float = 0.25
    ) -> Dict[str, Any]:
        """
        Runs dual-head object detection and associates detections across frames with temporal smoothing.
        Returns annotated frame, active track list, and any newly triggered alerts.
        """
        if frame is None:
            return {"annotated_frame": frame, "tracks": [], "alerts": []}

        # 1. Run Ensemble Detection (Stationery + Unauthorized Devices)
        detections = self.detector.detect(frame, conf=conf)

        # 2. Associate Detections with Active Tracks using IoU
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_track_ids = list(self.active_tracks.keys())

        if self.active_tracks and detections:
            # Build cost matrix
            iou_matrix = np.zeros((len(detections), len(unmatched_track_ids)))
            for d_idx, det in enumerate(detections):
                for t_idx, trk_id in enumerate(unmatched_track_ids):
                    trk_box = self.active_tracks[trk_id]["bbox"]
                    # Bonus for same class name
                    class_match = (det["class_name"] == self.active_tracks[trk_id]["class_name"])
                    iou = compute_iou(det["bbox"], trk_box)
                    if class_match:
                        iou += 0.15
                    iou_matrix[d_idx, t_idx] = iou

            # Greedy bipartite matching
            used_detections = set()
            used_tracks = set()

            # Find matches above IoU threshold 0.25
            sorted_indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape))[0]

            for d_idx, t_idx in sorted_indices:
                if d_idx in used_detections or t_idx in used_tracks:
                    continue
                if iou_matrix[d_idx, t_idx] < 0.25:
                    break

                trk_id = unmatched_track_ids[t_idx]
                det = detections[d_idx]

                # Update existing track
                self.active_tracks[trk_id]["bbox"] = det["bbox"]
                self.active_tracks[trk_id]["confidence"] = det["confidence"]
                self.active_tracks[trk_id]["frames_seen"] += 1
                self.active_tracks[trk_id]["lost_frames"] = 0

                used_detections.add(d_idx)
                used_tracks.add(t_idx)

                det_with_track = dict(det)
                det_with_track["track_id"] = trk_id
                matched_tracks.append(det_with_track)

            unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
            unmatched_track_ids = [unmatched_track_ids[i] for i in range(len(unmatched_track_ids)) if i not in used_tracks]

        # 3. Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[new_id] = {
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "label": det["label"],
                "severity": det["severity"],
                "confidence": det["confidence"],
                "color": det["color"],
                "description": det.get("description", ""),
                "frames_seen": 1,
                "lost_frames": 0
            }

            det_with_track = dict(det)
            det_with_track["track_id"] = new_id
            matched_tracks.append(det_with_track)

        # 4. Handle lost tracks (smoothing: keep showing temporarily if missed for a frame or two)
        for trk_id in list(unmatched_track_ids):
            self.active_tracks[trk_id]["lost_frames"] += 1
            if self.active_tracks[trk_id]["lost_frames"] > self.max_lost_frames:
                del self.active_tracks[trk_id]
            elif self.active_tracks[trk_id]["frames_seen"] >= 3:
                # Add smoothed track to visualization so box doesn't flicker away
                lost_trk = self.active_tracks[trk_id]
                matched_tracks.append({
                    "track_id": trk_id,
                    "class_name": lost_trk["class_name"],
                    "label": lost_trk["label"],
                    "severity": lost_trk["severity"],
                    "confidence": lost_trk["confidence"],
                    "bbox": lost_trk["bbox"],
                    "color": lost_trk["color"],
                    "description": lost_trk.get("description", "")
                })

        # 5. Filter confirmed tracks for alert triggers
        confirmed_alerts = []
        for t in matched_tracks:
            trk_id = t.get("track_id")
            frames_seen = self.active_tracks.get(trk_id, {}).get("frames_seen", 1)
            if frames_seen >= MIN_DETECTION_FRAMES:
                confirmed_alerts.append(t)

        # 6. Trigger alerts for prohibited items
        new_alerts = alert_manager.process_detections(frame, confirmed_alerts, camera_id)

        # 7. Draw boxes and HUD
        annotated_frame = self.detector.annotate(frame, detections=[], tracks=matched_tracks)

        return {
            "annotated_frame": annotated_frame,
            "tracks": matched_tracks,
            "alerts": new_alerts
        }
