# ExamGuard: College Viva & Defense Guide

This guide contains the most critical conceptual and technical questions college examiners and professors ask during B.Tech project reviews and viva voce.

---

### Q1. Why did you choose YOLO over Faster R-CNN or SSD?
* **Answer:** Faster R-CNN is a two-stage detector (Region Proposal Network followed by classification). While accurate, it has a high inference latency ($5 \text{ to } 10 \text{ FPS}$ on typical hardware), making real-time multi-camera tracking difficult. 
* YOLO is a **single-stage detector** that frames object detection as a direct regression problem (predicting bounding box coordinates and class probabilities simultaneously in a single forward pass). It achieves real-time speeds ($30+ \text{ FPS}$) with comparable accuracy, which is essential for live CCTV monitoring.

---

### Q2. What is the difference between Object Detection and Object Tracking?
* **Answer:**
  * **Detection (YOLO):** Operates on a **frame-by-frame basis** in isolation. It detects bounding boxes and object classes at time $t$, but has no memory of whether that object existed at time $t-1$.
  * **Tracking (ByteTrack):** Maintains **temporal continuity across consecutive frames**. It assigns a unique persistent ID (e.g. `Track #12`) to each object, associates detections between frames using Kalman Filters and the Hungarian algorithm, and tracks the duration and movement of the object.

---

### Q3. Why did you use ByteTrack instead of DeepSORT?
* **Answer:** DeepSORT relies on a separate Deep Learning Re-ID (appearance feature extraction) network, which adds significant computational overhead. 
* **ByteTrack** utilizes simple, robust motion prediction (Kalman Filter) and a two-stage matching strategy that pairs not only high-confidence detections but also **low-confidence detections** to existing tracks. This prevents tracks from being lost or fragmented when an object is momentarily occluded or blurred (e.g. a student moving a pen or phone under a desk).

---

### Q4. Explain Precision, Recall, and mAP in simple terms for an exam hall.
* **Precision ($\frac{TP}{TP + FP}$):** "Out of all alerts ExamGuard generated, what percentage were real unauthorized items?" High precision means low false alarms.
* **Recall ($\frac{TP}{TP + FN}$):** "Out of all actual unauthorized items in the exam hall, what percentage did ExamGuard catch?" High recall means the system rarely misses a cheat item.
* **$\text{mAP}_{50}$ (Mean Average Precision at IoU 0.50):** The area under the Precision-Recall curve evaluated when the predicted bounding box overlaps the ground truth by at least 50%. It measures overall detection quality.

---

### Q5. Why does ExamGuard adopt a "Human-in-the-Loop" architecture instead of automated disciplinary action?
* **Answer:** In high-stakes college examinations, false accusations can lead to severe emotional distress, legal complications, and academic unfairness. 
* Computer vision is susceptible to edge cases (e.g., a dark rectangular calculator case resembling a smartphone, or an innocent scratch paper resembling a cheat sheet). 
* Therefore, **ExamGuard treats AI as an assistant, not a judge**. AI identifies potential infractions, captures time-stamped evidence, and alerts the human invigilator, who makes the final verified determination.

---

### Q6. How does ExamGuard prevent alert spamming?
* **Answer:**
  1. **Temporal Persistence Threshold:** An object must be detected in at least 3 consecutive frames (`MIN_DETECTION_FRAMES = 3`) before triggering an alert, filtering out 1-frame optical glitches.
  2. **Alert Cooldown System:** Once an alert is triggered for a specific object/track ID, a 3-second cooldown window prevents duplicate alerts while the object remains in the camera's view.
  3. **Configurable Confidence Threshold:** Invigilators can adjust detection sensitivity (e.g. from 25% to 50%) directly from the web dashboard in real time.

---

### Q7. What is the database schema and why SQLite?
* **Answer:** SQLite was chosen for zero-configuration, self-contained, ACID-compliant local deployment without requiring complex external database server setup on invigilator workstations. 
* The `alerts` table records: `alert_id`, `timestamp`, `camera_id`, `object_class`, `confidence`, `severity`, `bbox`, `evidence_path`, `review_status` (PENDING / FLAGGED / DISMISSED), and `invigilator_notes`.

