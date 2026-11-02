"""
ExamGuard - Sample Video Generator
Creates a sample video test clip in videos/sample_exam.mp4 for testing
the detection pipeline and tracking when a physical camera is not connected.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VIDEOS_DIR

def create_sample_video(output_path: Path = VIDEOS_DIR / "sample_exam.mp4", duration_sec: int = 10, fps: int = 20):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    for i in range(total_frames):
        # Create exam desk background
        frame = np.full((height, width, 3), (230, 225, 215), dtype=np.uint8)

        # Desk surface
        cv2.rectangle(frame, (40, 80), (600, 440), (200, 190, 175), -1)
        cv2.rectangle(frame, (40, 80), (600, 440), (160, 150, 135), 2)

        # Desk Header Text
        cv2.putText(frame, "Desk #14 - Candidate View", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

        # Draw Allowed Stationery: Pencil
        cv2.rectangle(frame, (100, 200), (115, 340), (40, 180, 220), -1)
        cv2.putText(frame, "Pencil", (85, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        # Draw Allowed Stationery: Eraser
        cv2.rectangle(frame, (160, 240), (210, 280), (240, 240, 240), -1)
        cv2.rectangle(frame, (160, 240), (210, 280), (180, 180, 180), 1)
        cv2.putText(frame, "Eraser", (162, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        # Draw Allowed Exam Paper Sheet
        cv2.rectangle(frame, (250, 150), (480, 410), (255, 255, 255), -1)
        cv2.rectangle(frame, (250, 150), (480, 410), (180, 180, 180), 1)
        cv2.putText(frame, "EXAMINATION QUESTION PAPER", (265, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        for line_y in range(210, 390, 25):
            cv2.line(frame, (270, line_y), (460, line_y), (220, 220, 220), 1)

        # Simulated Incident: Mobile Phone appears on frame 40 onwards
        if i >= 40:
            # Moving / placed phone
            phone_x = min(350, 200 + (i - 40) * 3)
            phone_y = 230
            # Phone body
            cv2.rectangle(frame, (phone_x, phone_y), (phone_x + 65, phone_y + 115), (30, 30, 30), -1)
            # Screen
            cv2.rectangle(frame, (phone_x + 4, phone_y + 10), (phone_x + 61, phone_y + 105), (10, 80, 160), -1)
            cv2.putText(frame, "Cell Phone", (phone_x, phone_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        # Time overlay
        cv2.putText(frame, f"Time: 10:15:{i//fps:02d} AM", (470, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        out.write(frame)

    out.release()
    print(f"[+] Sample test exam video generated at: {output_path}")

if __name__ == "__main__":
    create_sample_video()
