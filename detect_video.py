"""
ExamGuard - Standalone Video Object Detection & Tracking Tool
Run this script with any video file to detect objects, track them across frames,
and generate a comprehensive detection report with saved evidence and annotated video.

Usage:
    python detect_video.py --video path/to/video.mp4
    python detect_video.py --video 0  (for live webcam)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RUNS_DIR, EVIDENCE_DIR, VIDEOS_DIR
from src.detector import Detector
from src.tracker import ObjectTracker
from src.alerts import alert_manager
from src.database import db

def process_video(
    video_source: str,
    conf_threshold: float = 0.25,
    save_output: bool = True,
    show_window: bool = False
):
    print("=" * 65)
    print("         EXAMGUARD: VIDEO OBJECT DETECTION & TRACKING")
    print("=" * 65)
    print(f" Source Video: {video_source}")
    print(f" Confidence:   {int(conf_threshold * 100)}%")
    print(f" Tracker:      ByteTrack Multi-Object Tracking")
    print("=" * 65)

    # Resolve video source (digit for webcam or file path)
    if video_source.isdigit():
        src = int(video_source)
        is_webcam = True
        video_name = f"webcam_{src}"
    else:
        src = str(video_source)
        is_webcam = False
        video_name = Path(src).stem
        if not os.path.exists(src):
            print(f"[!] Error: Video file not found: {src}")
            return

    # Initialize Video Capture
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {src}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    print(f" Video Resolution: {width}x{height} @ {int(fps)} FPS")
    if not is_webcam:
        duration_sec = total_frames / fps
        print(f" Total Frames:     {total_frames} (~{duration_sec:.1f} seconds)")

    # Initialize Detector & Tracker
    print("\n[+] Initializing AI Detection & Tracking Engine...")
    detector = Detector()
    tracker = ObjectTracker(detector)

    # Setup Video Writer if saving
    out_writer = None
    output_video_path = None
    if save_output:
        save_dir = RUNS_DIR / "detect"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = save_dir / f"detected_{video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"[+] Output annotated video will be saved to:\n    {output_video_path}")

    # Detection Statistics Collectors
    detected_objects_summary = defaultdict(int)
    object_timeline = []  # list of {"time": ..., "label": ..., "track_id": ..., "conf": ..., "severity": ...}
    seen_track_keys = set()
    total_alerts_triggered = 0

    frame_idx = 0
    start_time = time.time()

    print("\n[+] Processing Video Frames...\n")
    pbar = tqdm(total=total_frames if not is_webcam else None, unit="frames", desc="ExamGuard Analyzing")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp_sec = frame_idx / fps
            timestamp_str = time.strftime('%H:%M:%S', time.gmtime(timestamp_sec))

            # Run ByteTrack Tracking & Object Detection
            tracking_result = tracker.track_frame(
                frame=frame,
                camera_id=video_name,
                conf=conf_threshold
            )

            annotated_frame = tracking_result["annotated_frame"]
            active_tracks = tracking_result["tracks"]
            new_alerts = tracking_result["alerts"]

            # Log new alerts
            total_alerts_triggered += len(new_alerts)

            # Record newly seen objects for timeline report
            for t in active_tracks:
                track_id = t.get("track_id")
                label = t["label"]
                severity = t["severity"]
                conf = t["confidence"]
                key = f"{label}_{track_id}" if track_id is not None else f"{label}_{frame_idx}"

                if key not in seen_track_keys:
                    seen_track_keys.add(key)
                    detected_objects_summary[label] += 1
                    object_timeline.append({
                        "frame": frame_idx,
                        "time": timestamp_str,
                        "label": label,
                        "severity": severity,
                        "conf": conf,
                        "track_id": track_id
                    })

            # Write to output video
            if out_writer is not None:
                out_writer.write(annotated_frame)

            # Display window if requested
            if show_window:
                cv2.imshow("ExamGuard Live Analysis", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[!] User interrupted video playback.")
                    break

            pbar.update(1)

    except KeyboardInterrupt:
        print("\n[!] Processing stopped by user.")
    finally:
        pbar.close()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if show_window:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0

    # -------------------------------------------------------------
    # PRINT SUMMARY REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("                EXAMGUARD DETECTION REPORT")
    print("=" * 65)
    print(f" Frames Processed:   {frame_idx} frames")
    print(f" Processing Time:    {elapsed:.2f} seconds ({proc_fps:.1f} FPS)")
    print(f" Total Alerts:       {total_alerts_triggered} incident alerts generated")
    print("-" * 65)
    print(" DETECTED OBJECT BREAKDOWN:")
    print("-" * 65)

    if not detected_objects_summary:
        print("  * No exam objects detected at the current confidence threshold.")
    else:
        for label, count in sorted(detected_objects_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {label:<22} : {count} unique instance(s)")

    print("-" * 65)
    print(" DETECTION TIMELINE (First Appearance Log):")
    print("-" * 65)
    if not object_timeline:
        print("  * No timeline events recorded.")
    else:
        print(f"  {'Time':<10} {'Severity':<10} {'Object':<20} {'Conf':<8} {'Track ID'}")
        print("  " + "-" * 58)
        for event in object_timeline[:25]:  # Display up to 25 timeline events
            track_str = f"#{event['track_id']}" if event['track_id'] is not None else "-"
            print(f"  {event['time']:<10} {event['severity']:<10} {event['label']:<20} {int(event['conf']*100)}%    {track_str}")
        if len(object_timeline) > 25:
            print(f"  ... and {len(object_timeline) - 25} more items.")

    print("-" * 65)
    if output_video_path and output_video_path.exists():
        print(f"\n[+] Annotated Video Saved:")
        print(f"    file:///{output_video_path.resolve()}")
    
    print(f"[+] Evidence Snapshots Saved in:")
    print(f"    file:///{EVIDENCE_DIR.resolve()}")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="ExamGuard Video Detection Tool")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (.mp4, .avi) or webcam index (0)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.1 to 0.9, default 0.25)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV window during processing"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output annotated video"
    )
    args = parser.parse_args()

    # Determine video source
    video_source = args.video
    if not video_source:
        # Check if sample video exists
        sample_path = VIDEOS_DIR / "sample_exam.mp4"
        if sample_path.exists():
            video_source = str(sample_path)
            print(f"[*] No video path provided. Using default: {video_source}")
        else:
            # Prompt user
            video_source = input("Enter path to exam video (or 0 for webcam): ").strip()

    process_video(
        video_source=video_source,
        conf_threshold=args.conf,
        save_output=not args.no_save,
        show_window=args.show
    )

if __name__ == "__main__":
    main()

