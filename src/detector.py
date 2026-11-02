"""
ExamGuard - Dual-Head Ensemble YOLO Detection Engine
Combines:
  1. Custom Stationery Weights (best.pt): Precision-trained on Pencil, Eraser, Sharpener, Ruler.
  2. High-Capacity YOLO11s Weights: Accurately detects Mobile Phones, Books, Laptops, Headphones.
Merges detections via Non-Maximum Suppression (NMS) for zero-conflict, comprehensive monitoring.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from src.config import (
    CLASS_CONFIG,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_CUSTOM_MODEL,
    DEFAULT_PRETRAINED_MODEL
)

def compute_iou(boxA, boxB) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class Detector:
    def __init__(self, model_path: Optional[str] = None):
        self.custom_model = None
        self.threat_model = None
        self.is_custom_model = False
        self.model = None  # Primary model for ByteTrack tracking API
        self._load_engines(model_path)

    def _load_engines(self, custom_path: Optional[str] = None):
        """Loads both specialized models for ensemble monitoring."""
        from ultralytics import YOLO

        # 1. Load Threat Detection Engine (YOLO11s)
        threat_path = str(DEFAULT_PRETRAINED_MODEL)
        try:
            print(f"[ExamGuard Detector] Loading Threat Detection Engine from: {threat_path}")
            self.threat_model = YOLO(threat_path)
            self.model = self.threat_model  # Primary tracker reference
        except Exception as e:
            print(f"[!] Warning: Could not load threat model: {e}")

        # 2. Load Custom Stationery Engine (best.pt)
        custom_weights = custom_path if (custom_path and os.path.exists(custom_path)) else str(DEFAULT_CUSTOM_MODEL)
        if os.path.exists(custom_weights):
            try:
                print(f"[ExamGuard Detector] Loading Custom Stationery Engine from: {custom_weights}")
                self.custom_model = YOLO(custom_weights)
                self.is_custom_model = True
                print(f"[ExamGuard Detector] Active Custom Classes: {list(self.custom_model.names.values())}")
            except Exception as e:
                print(f"[!] Warning: Could not load custom stationery model: {e}")
        else:
            print("[ExamGuard Detector] Custom stationery model not found. Running in Pretrained mode.")

    @property
    def model_path(self) -> str:
        if self.custom_model and self.threat_model:
            return "Dual-Head Ensemble (YOLO11s + Custom best.pt)"
        elif self.custom_model:
            return str(DEFAULT_CUSTOM_MODEL)
        return str(DEFAULT_PRETRAINED_MODEL)

    def detect(self, frame: np.ndarray, conf: float = DEFAULT_CONF_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Runs dual-head detection on frame and merges results.
        Stationery is extracted from custom model; unauthorized devices from threat model.
        """
        if frame is None:
            return []

        all_detections = []

        # Head 1: Run Threat Detector (Phones, Books, Laptops, etc.)
        if self.threat_model is not None:
            threat_results = self.threat_model.predict(
                source=frame,
                conf=conf,
                iou=DEFAULT_IOU_THRESHOLD,
                verbose=False
            )
            if threat_results and len(threat_results) > 0 and threat_results[0].boxes is not None:
                for box in threat_results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = self.threat_model.names.get(cls_id, "").lower().strip()
                    mapped = self._map_class_name(raw_name)
                    # Threat model only handles prohibited / suspicious objects
                    if mapped in ["cell phone", "laptop", "book", "headphones"]:
                        config = CLASS_CONFIG.get(mapped)
                        if config:
                            all_detections.append({
                                "class_name": mapped,
                                "label": config["label"],
                                "severity": config["severity"],
                                "confidence": round(float(box.conf[0].item()), 3),
                                "bbox": [int(v) for v in box.xyxy[0].tolist()],
                                "color": config["color"],
                                "description": config["description"]
                            })

        # Head 2: Run Custom Stationery Detector (Pencils, Erasers, Sharpeners, Rulers)
        if self.custom_model is not None:
            stat_results = self.custom_model.predict(
                source=frame,
                conf=max(0.28, conf),  # Stable threshold for stationery to prevent false alarms
                iou=DEFAULT_IOU_THRESHOLD,
                verbose=False
            )
            if stat_results and len(stat_results) > 0 and stat_results[0].boxes is not None:
                for box in stat_results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = self.custom_model.names.get(cls_id, "").lower().strip()
                    mapped = self._map_class_name(raw_name)
                    config = CLASS_CONFIG.get(mapped)
                    if config:
                        all_detections.append({
                            "class_name": mapped,
                            "label": config["label"],
                            "severity": config["severity"],
                            "confidence": round(float(box.conf[0].item()), 3),
                            "bbox": [int(v) for v in box.xyxy[0].tolist()],
                            "color": config["color"],
                            "description": config["description"]
                        })

        # Apply Non-Maximum Suppression across both heads to resolve overlaps
        filtered_detections = self._apply_nms(all_detections, iou_thresh=0.50)
        return filtered_detections

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.50) -> List[Dict[str, Any]]:
        """Removes lower confidence duplicate boxes covering the same object."""
        if len(detections) <= 1:
            return detections

        # Sort descending by confidence
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = []

        while len(sorted_dets) > 0:
            best = sorted_dets.pop(0)
            keep.append(best)

            survivors = []
            for candidate in sorted_dets:
                iou = compute_iou(best["bbox"], candidate["bbox"])
                if iou < iou_thresh:
                    survivors.append(candidate)
            sorted_dets = survivors

        return keep

    def _map_class_name(self, raw_name: str) -> str:
        """Map standard COCO and custom names into standard ExamGuard names."""
        name = raw_name.replace(" ", "_")
        if name in ["cell_phone", "mobile_phone", "phone"]:
            return "cell phone"
        if name in ["laptop", "computer"]:
            return "laptop"
        if name in ["book"]:
            return "book"
        if name in ["pencil"]:
            return "pencil"
        if name in ["eraser"]:
            return "eraser"
        if name in ["sharpener"]:
            return "sharpener"
        if name in ["ruler"]:
            return "ruler"
        return raw_name

    def annotate(self, frame: np.ndarray, detections: List[Dict[str, Any]], tracks: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """Draws aesthetic bounding boxes, badges, and alerts on the frame."""
        annotated = frame.copy()
        items = tracks if tracks is not None else detections

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            color = item.get("color", (0, 255, 0))
            label = item.get("label", item.get("class_name", "Object"))
            conf = item.get("confidence", 0.0)
            track_id = item.get("track_id")
            severity = item.get("severity", "INFO")

            # Draw bounding box with rounded glow
            thickness = 3 if severity == "HIGH" else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            tag = f"{label} {int(conf * 100)}%"
            if track_id is not None:
                tag = f"#{track_id} {tag}"

            # Label badge background
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + w + 12, max(24, y1)), color, -1)
            
            # Label text
            text_color = (255, 255, 255) if severity in ["HIGH", "MEDIUM"] else (0, 0, 0)
            cv2.putText(
                annotated,
                tag,
                (x1 + 6, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA
            )

        return annotated
