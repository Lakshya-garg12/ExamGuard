# ExamGuard: System Architecture & Workflow

ExamGuard is designed as an AI-assisted monitoring system that enhances examination hall integrity by detecting prohibited objects and assisting human invigilators.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph VideoSource["1. Video Ingestion Layer"]
        A1["Exam Hall IP Camera / RTSP"] --> B["Video Stream Manager"]
        A2["USB Web Camera (Invigilator Desk)"] --> B
        A3["Recorded Exam Video (.mp4 / .avi)"] --> B
    end

    subgraph PerceptionLayer["2. AI Perception & Tracking Layer"]
        B --> C["YOLO Object Detection Engine (YOLO11 / Custom Model)"]
        C --> D["ByteTrack Multi-Object Tracker"]
        D --> E["Bounding Box Normalization & Filtering"]
    end

    subgraph ReasoningLayer["3. Decision & Alert Engine"]
        E --> F{"Object Severity Classifier"}
        F -->|Permitted Items: Pencil, Eraser, Ruler| G["Permitted Log (Severity: INFO)"]
        F -->|Prohibited Items: Phone, Book, Notes| H["Incident Verification Filter"]
        H --> I["Persistence & Cooldown Check (>= 3 frames)"]
        I --> J["Generate Alert Record (ALT-YYYYMMDD-ID)"]
        J --> K["Capture Visual Evidence Snapshot (JPEG)"]
    end

    subgraph StorageLayer["4. Persistence & Database"]
        J --> L[("SQLite Database (examguard.db)")]
        K --> M["Evidence Storage (/data/alerts_evidence/)"]
    end

    subgraph ApplicationLayer["5. Invigilator Interface"]
        L --> N["FastAPI REST & Streaming Server"]
        M --> N
        B --> N
        N --> O["Web Invigilator Dashboard (HTML5 / Tailwind)"]
        O --> P["Live Video HUD with Tracking Boxes"]
        O --> Q["Real-Time Alert Feed & Incident Queue"]
        O --> R["Human-in-the-Loop Review & Notes Modal"]
        R -->|Confirmed Infraction or False Alarm| L
    end
```

---

## 2. Multi-Object Tracking Pipeline (ByteTrack)

```mermaid
sequenceDiagram
    autonumber
    participant Camera as Camera Stream
    participant Detector as YOLO Detector
    participant Tracker as ByteTrack Tracker
    participant Alerts as Alert Manager
    participant DB as SQLite DB
    participant UI as Invigilator Dashboard

    Camera->>Detector: New Frame (Frame #T)
    Detector->>Tracker: Detections (Boxes + Class IDs + Confidence)
    Tracker->>Tracker: Associate Detections with Kalman Filter Tracks
    Tracker->>Alerts: Active Track (#ID, Class, Frames Visible)
    
    alt Unauthorized Object Visible >= 3 Frames
        Alerts->>Alerts: Check Alert Cooldown (3 seconds)
        Alerts->>Alerts: Save Annotated Evidence Snapshot
        Alerts->>DB: Insert Alert (PENDING Status)
        Alerts->>UI: Broadcast New Incident Alert
        UI->>UI: Play Visual / Audio Alert
    else Stationery Item or Below Threshold
        Alerts->>UI: Update Live HUD Only
    end

    Note over UI: Human Invigilator Reviews Alert
    UI->>DB: Update Review: FLAGGED_INFRACTION or DISMISSED
```

---

## 3. Human-in-the-Loop Review Philosophy

A core tenet of ExamGuard is that **AI never accuses a student directly**:
* AI is purely an anomaly detection sensor.
* Detections create an **incident alert** with a time-stamped evidence frame.
* The human invigilator inspects the evidence, physically verifies the desk, and records their decision and remarks in the audit log.

