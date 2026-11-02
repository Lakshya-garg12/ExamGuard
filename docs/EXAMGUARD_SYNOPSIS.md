# Project Synopsis: ExamGuard
## AI-Based Exam Environment Monitoring System

* **Project Title:** ExamGuard: AI-Based Exam Environment Monitoring System
* **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)
* **Domain:** Deep Learning, Computer Vision, Real-Time Video Analytics
* **Technologies:** Python, YOLO (You Only Look Once), OpenCV, PyTrack/ByteTrack, FastAPI, SQLite, Tailwind CSS

---

### 1. Abstract
Maintaining academic integrity in examination halls is a vital yet challenging operational task. Human invigilators often oversee dozens of students simultaneously, leading to human fatigue, blind spots, and missed unauthorized items such as concealed mobile phones, cheat notes, or textbooks. 

**ExamGuard** is an assistive AI computer vision framework designed to monitor examination environments in real time. Rather than replacing the invigilator or making automated accusations, ExamGuard functions as an intelligent assistant. It continuously processes video streams from exam hall cameras, detects both permitted stationery and unauthorized objects using state-of-the-art YOLO object detectors, tracks objects across consecutive frames using ByteTrack, and generates time-stamped incident alerts with captured visual evidence frames. A centralized web-based dashboard allows human invigilators to review flagged incidents, verify situations on the ground, and record formal audit notes.

---

### 2. Problem Statement
Traditional examination monitoring suffers from significant vulnerabilities:
1. **Human Fatigue & Divided Attention:** A single invigilator supervising 30 to 60 candidates cannot continuously observe every desk simultaneously.
2. **Concealment of Smart Devices:** The proliferation of miniature smartphones, smartwatches, and wireless earphones has made physical cheating harder to catch with casual observation.
3. **Absence of Objective Audit Trails:** When cheating is suspected, there is often a lack of indisputable photographic or temporal evidence to support administrative disciplinary actions.
4. **Risk of Bias:** Automated systems that unilaterally penalize students create high false-alarm rates and ethical risks.

---

### 3. Objectives
1. **Automated Object Detection:** Detect prohibited items (mobile phones, textbooks, loose sheets) and distinguish them from permitted stationery (pencils, erasers, sharpeners, rulers).
2. **Temporal Object Tracking:** Associate detected items across frames using ByteTrack to filter out transient false alarms and evaluate object dwell time.
3. **Evidence Capture & Storage:** Automatically extract, annotate, and persist high-resolution snapshot evidence with timestamps and bounding box coordinates into an audit database.
4. **Human-in-the-Loop Review System:** Provide an intuitive dashboard enabling human invigilators to confirm infractions or dismiss false alarms with observational notes.
5. **Modular Architecture:** Build an extensible pipeline supporting live RTSP camera feeds, webcams, and pre-recorded video files.

---

### 4. Existing System vs. Proposed System

| Parameter | Traditional Manual Invigilation | Existing CCTV Systems | Proposed ExamGuard System |
|---|---|---|---|
| **Monitoring Mechanism** | Visual observation by human teacher | Passive passive recording | Active real-time AI perception + Human Review |
| **Alert Response Time** | Delayed (depends on noticing) | Post-exam manual playback | Instantaneous (sub-second alert generation) |
| **Object Classification** | Subjective | None | Automated multi-class YOLO detection |
| **Evidence Logging** | Manual paper reports | Huge continuous video search | Discrete, indexed evidence snapshots in DB |
| **Decision Authority** | Human | Human (manual review) | **Human Invigilator (AI as Assistant)** |

---

### 5. System Requirements

#### Hardware Requirements:
* **Processor:** Intel Core i5 / AMD Ryzen 5 or higher
* **RAM:** 8 GB minimum (16 GB recommended)
* **Storage:** 5 GB free disk space
* **Camera:** Standard 720p / 1080p USB Webcam or IP Camera
* **GPU (Optional):** NVIDIA GTX 1650 / RTX 3050 or higher for GPU acceleration

#### Software Requirements:
* **Operating System:** Windows 10/11 or Ubuntu Linux
* **Language:** Python 3.10 / 3.11
* **Computer Vision & Deep Learning:** Ultralytics YOLO, PyTorch, OpenCV, Lapx (ByteTrack)
* **Backend Framework:** FastAPI, Uvicorn, SQLite3
* **Frontend Interface:** HTML5, Tailwind CSS, JavaScript

---

### 6. Methodology & Core Modules

1. **Dataset Engineering Module (`src/prepare_dataset.py`):**
   * Prepares and validates normalized YOLO bounding boxes ($x, y, w, h \in [0.0, 1.0]$).
   * Enforces a unified Master Class Mapping to prevent ID collision across disparate data sources.
   * Splits data into standard 70% Train, 20% Validation, and 10% Test distributions.

2. **Perception & Tracking Engine (`src/detector.py`, `src/tracker.py`):**
   * Employs YOLO for single-stage object localization and classification.
   * Leverages ByteTrack with Kalman filtering to track persistent bounding boxes across frames.
   * Applies severity filters (High: Phones/Laptops; Medium: Books/Paper; Info: Stationery).

3. **Alert & Evidence Manager (`src/alerts.py`, `src/database.py`):**
   * Implements temporal persistence: an object must appear for $\ge 3$ frames to prevent momentary noise triggers.
   * Enforces alert cooldowns (3 seconds) to avoid spamming the invigilator.
   * Encodes and writes visual snapshot evidence into `data/alerts_evidence/`.
   * Records alerts into SQLite table with review status `PENDING`.

4. **Invigilator Web Dashboard (`frontend/index.html`, `src/server.py`):**
   * Live MJPEG video stream with real-time HUD and detection boxes.
   * Dynamic incident alert stream with severity badges.
   * One-click "Inspect & Review" modal presenting evidence snapshots, metadata, and review actions (`FLAGGED_INFRACTION` vs `DISMISSED_FALSE_ALARM`).

---

### 7. Preliminary Results (Test Evaluation)
From preliminary testing on the stationery dataset (34 test images, 4 classes: Pencil, Eraser, Sharpener, Ruler):

* **Precision:** 90.84%
* **Recall:** 74.13%
* **$\text{mAP}_{50}$:** 82.75%
* **$\text{mAP}_{50-95}$:** 59.65%

---

### 8. Conclusion
ExamGuard demonstrates a practical, ethically sound implementation of AI in academic supervision. By augmenting invigilators with automated detection and real-time alerts without taking autonomous disciplinary actions, the system bridges the gap between passive video surveillance and human inspection.

