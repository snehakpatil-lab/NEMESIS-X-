# NEMESIS-X backend (Stage 2)

FastAPI + YOLO/ByteTrack + SQLite. Runs fully local; **no API key needed**.

## Setup
1. Python 3.10+.
2. Install PyTorch first (CUDA build for RTX 3050/4050): https://pytorch.org/get-started/locally/
   (CPU-only also works, just slower.)
3. Then:
   ```
   python -m venv .venv
   # Windows: .venv\Scripts\activate      Ubuntu: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --port 8000
   ```
4. Open http://localhost:8000 (first analysis downloads the YOLO weights, so it needs internet once).

## Config (env vars)
NEMESIS_MODEL (default yolov8n.pt) · NEMESIS_CONF (0.25) · NEMESIS_STRIDE (minimum 1) · NEMESIS_MAX_ANALYSIS_FRAMES (700) · NEMESIS_IMGSZ (416)

## API
GET /api/health · /api/system · /api/system/configuration · POST /api/videos · GET /api/videos · GET /api/videos/{id}
POST /api/videos/{id}/analyze · GET .../tracks · .../tracks/{tid} · .../detections?t=SEC
GET .../events · .../memory?offset=0&limit=50 · .../spatial · .../frame?t=SEC · .../file
GET .../anomalies · POST .../anomalies/{candidate_id}/review · GET .../evidence?offset=0&limit=30
POST /api/chat {question, video_id?}

## Notes
- Ultralytics is AGPL-3.0; check its licence for your use.
- `/api/system` reports CPU/RAM use, CUDA/GPU telemetry, and processing state/FPS; VRAM is not applicable during CPU-only inference.
- Video inference samples at most 700 frames by default and uses 416px model input; the effective stride is saved per video. This speeds up long clips at the cost of temporal detail between samples.
- Coordinates are image-plane pixels, not metric 3D.
- Track event records mark first/last observations only; they are not incident classifications.
- Statistical anomaly candidates are robust within-class high-speed outliers in image-plane tracking and require human review; camera motion or tracker switches can cause false positives.
- Model-generated candidates come from a deterministic Isolation Forest fitted per video and object class using track speed, path efficiency, detector confidence, and observation count. Scores are not probabilities; classes need at least 12 qualifying tracks.
- Candidate review is persisted per video; neither statistical nor model-generated candidates are automatically treated as verified events.
- Evidence Explorer items are paginated summaries of stored detection boxes with first/last timestamps, source-frame links, reasoning basis, and explicit data limitations.
- Evidence Q&A summarizes tracked classes and timeline activity, and links vehicle motion outliers to reviewable timestamps. These image-plane cues are not collision detections and cannot establish impact, injury, or cause.
- Semantic event/incident classification and cross-video re-identification remain unimplemented.
