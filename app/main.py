import math, re, shutil, threading, uuid
from collections import defaultdict
from statistics import median
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from . import db, pipeline, system
from .config import ALLOWED, ROOT, UPLOADS

app = FastAPI(title="NEMESIS-X")
db.init()
_lock = threading.Lock()   # one analysis at a time (GPU friendly)

def video(c, vid):
    r = c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    if not r: raise HTTPException(404, "Unknown video")
    return r

@app.get("/api/health")
def health():
    return {"status": "ok", "device": "cuda" if pipeline.device() == 0 else "cpu"}

@app.get("/api/system")
def sys_metrics(): return system.snapshot()

@app.get("/api/system/configuration")
def sys_configuration(): return system.configuration()

@app.post("/api/videos")
def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400, f"Unsupported type {ext!r}; use {sorted(ALLOWED)}")
    dest = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
    c = db.conn(); cur = c.execute("INSERT INTO videos(name,path) VALUES(?,?)", (file.filename, str(dest))); c.commit()
    return {"id": cur.lastrowid, "name": file.filename}

@app.get("/api/videos")
def list_videos():
    c = db.conn(); return [dict(r) for r in c.execute(
        "SELECT id,name,status,stage,progress,duration,analysis_stride FROM videos ORDER BY id DESC"
    )]

@app.get("/api/videos/{vid}")
def status(vid: int): return {k: v for k, v in dict(video(db.conn(), vid)).items() if k != "path"}

@app.get("/api/videos/{vid}/file")
def file(vid: int): return FileResponse(video(db.conn(), vid)["path"])

@app.post("/api/videos/{vid}/analyze")
def analyze(vid: int):
    video(db.conn(), vid)
    def job():
        with _lock: pipeline.run(vid)
    threading.Thread(target=job, daemon=True).start()
    return {"queued": vid}

@app.get("/api/videos/{vid}/tracks")
def tracks(vid: int):
    c = db.conn(); video(c, vid)
    q = """SELECT track_id id, cls, COUNT(*) n, MIN(t) first_t, MAX(t) last_t, AVG(conf) mean_conf
           FROM detections WHERE video_id=? GROUP BY track_id, cls ORDER BY track_id"""
    return [dict(r) for r in c.execute(q, (vid,))]

@app.get("/api/videos/{vid}/tracks/{tid}")
def track(vid: int, tid: int):
    c = db.conn(); video(c, vid)
    rows = c.execute("SELECT frame,t,cls,conf,x1,y1,x2,y2 FROM detections WHERE video_id=? AND track_id=? ORDER BY frame", (vid, tid)).fetchall()
    if not rows: raise HTTPException(404, "No such track")
    return {"id": tid, "coordinate_space": "image-plane pixels (not metric 3D)", "points": [dict(r) for r in rows]}

@app.get("/api/videos/{vid}/events")
def events(vid: int):
    c = db.conn(); video(c, vid)
    tracks = c.execute(
        "SELECT track_id,cls,COUNT(*) observations,MIN(t) first_t,MAX(t) last_t "
        "FROM detections WHERE video_id=? GROUP BY track_id,cls ORDER BY first_t,track_id",
        (vid,),
    ).fetchall()
    result = []
    for row in tracks:
        for kind, t in (("first_observed", row["first_t"]), ("last_observed", row["last_t"])):
            result.append({"type": kind, "t": t, "track_id": row["track_id"],
                           "cls": row["cls"], "observations": row["observations"]})
    return sorted(result, key=lambda event: (event["t"], event["track_id"], event["type"]))

@app.get("/api/videos/{vid}/memory")
def memory(vid: int, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    c = db.conn(); r = video(c, vid)
    total = c.execute("SELECT COUNT(*) FROM detections WHERE video_id=?", (vid,)).fetchone()[0]
    tracks = c.execute(
        "SELECT track_id id,cls,COUNT(*) observations,MIN(t) first_t,MAX(t) last_t "
        "FROM detections WHERE video_id=? GROUP BY track_id,cls ORDER BY observations DESC,track_id",
        (vid,),
    ).fetchall()
    observations = c.execute(
        "SELECT frame,t,track_id,cls,conf,x1,y1,x2,y2 FROM detections "
        "WHERE video_id=? ORDER BY t,track_id LIMIT ? OFFSET ?",
        (vid, limit, offset),
    ).fetchall()
    return {"video_id": vid, "duration": r["duration"], "total_observations": total,
            "total_tracks": len(tracks), "offset": offset, "limit": limit,
            "tracks": [dict(row) for row in tracks],
            "observations": [dict(row) for row in observations]}

@app.get("/api/videos/{vid}/spatial")
def spatial(vid: int, limit: int = Query(50, ge=1, le=100),
            points_per_track: int = Query(120, ge=2, le=300)):
    c = db.conn(); r = video(c, vid)
    total_tracks = c.execute(
        "SELECT COUNT(*) FROM (SELECT track_id,cls FROM detections WHERE video_id=? GROUP BY track_id,cls)",
        (vid,),
    ).fetchone()[0]
    summaries = c.execute(
        "SELECT track_id id,cls,COUNT(*) observations FROM detections WHERE video_id=? "
        "GROUP BY track_id,cls ORDER BY observations DESC,track_id LIMIT ?",
        (vid, limit),
    ).fetchall()
    result = []
    for summary in summaries:
        points = c.execute(
            "SELECT t,(x1+x2)/2.0 x,(y1+y2)/2.0 y FROM detections "
            "WHERE video_id=? AND track_id=? AND cls=? ORDER BY frame",
            (vid, summary["id"], summary["cls"]),
        ).fetchall()
        step = max(1, (len(points) + points_per_track - 1) // points_per_track)
        sampled = list(points[::step])
        if points and sampled[-1]["t"] != points[-1]["t"]:
            sampled.append(points[-1])
        result.append({"id": summary["id"], "cls": summary["cls"],
                       "observations": summary["observations"],
                       "points": [dict(point) for point in sampled]})
    return {"video_id": vid, "coordinate_space": "image-plane pixels (x right, y down; not metric 3D)",
            "width": r["width"], "height": r["height"], "total_tracks": total_tracks,
            "tracks_returned": len(result), "tracks": result}

def motion_anomalies(c, vid):
    rows = c.execute(
        "SELECT track_id,cls,t,conf,(x1+x2)/2.0 x,(y1+y2)/2.0 y FROM detections "
        "WHERE video_id=? ORDER BY track_id,cls,frame", (vid,),
    ).fetchall()
    groups = defaultdict(list)
    for row in rows:
        groups[(row["track_id"], row["cls"])].append(row)
    class_tracks = defaultdict(list)
    for (track_id, cls), points in groups.items():
        if len(points) < 8:
            continue
        segments = []
        for before, after in zip(points, points[1:]):
            dt = after["t"] - before["t"]
            if dt > 0:
                speed = math.hypot(after["x"] - before["x"], after["y"] - before["y"]) / dt
                segments.append((speed, before["t"], after["t"]))
        if segments:
            class_tracks[cls].append((track_id, len(points), median([item[0] for item in segments]), segments))
    candidates = []
    for cls, tracks in class_tracks.items():
        if len(tracks) < 8:
            continue
        speeds = [track[2] for track in tracks]
        center = median(speeds)
        mad = median([abs(speed - center) for speed in speeds])
        if mad <= 0:
            continue
        scale = 1.4826 * mad
        for track_id, observations, speed, segments in tracks:
            score = (speed - center) / scale
            if score < 3.5:
                continue
            peak_speed, start_t, end_t = max(segments, key=lambda item: item[0])
            slug = re.sub(r"[^a-z0-9]+", "-", cls.lower()).strip("-")
            candidates.append({
                "id": f"speed-{slug}-{track_id}",
                "type": "image_plane_speed_outlier",
                "track_id": track_id,
                "cls": cls,
                "start_t": start_t,
                "end_t": end_t,
                "observations": observations,
                "median_speed_px_s": speed,
                "peak_segment_px_s": peak_speed,
                "class_median_px_s": center,
                "class_mad_px_s": mad,
                "robust_z": score,
            })
    return sorted(candidates, key=lambda item: item["robust_z"], reverse=True)

def model_motion_anomalies(c, vid):
    rows = c.execute(
        "SELECT track_id,cls,t,conf,(x1+x2)/2.0 x,(y1+y2)/2.0 y FROM detections "
        "WHERE video_id=? ORDER BY track_id,cls,frame", (vid,),
    ).fetchall()
    groups = defaultdict(list)
    for row in rows:
        groups[(row["track_id"], row["cls"])].append(row)

    class_tracks = defaultdict(list)
    for (track_id, cls), points in groups.items():
        if len(points) < 8:
            continue
        segments = []
        for before, after in zip(points, points[1:]):
            dt = after["t"] - before["t"]
            if dt > 0:
                speed = math.hypot(after["x"] - before["x"], after["y"] - before["y"]) / dt
                segments.append((speed, before["t"], after["t"]))
        if not segments:
            continue
        path_length = sum(speed * (end_t - start_t) for speed, start_t, end_t in segments)
        displacement = math.hypot(points[-1]["x"] - points[0]["x"], points[-1]["y"] - points[0]["y"])
        confidence = [point["conf"] for point in points if point["conf"] is not None]
        features = [
            math.log1p(median([segment[0] for segment in segments])),
            math.log1p(float(np.percentile([segment[0] for segment in segments], 90))),
            displacement / path_length if path_length else 0.0,
            1.0 - (sum(confidence) / len(confidence) if confidence else 0.0),
            math.log1p(len(points)),
        ]
        slug = re.sub(r"[^a-z0-9]+", "-", cls.lower()).strip("-")
        class_tracks[cls].append({
            "id": f"iforest-{slug}-{track_id}",
            "track_id": track_id,
            "cls": cls,
            "observations": len(points),
            "start_t": min(segment[1] for segment in segments),
            "end_t": max(segment[2] for segment in segments),
            "peak_segment_px_s": max(segment[0] for segment in segments),
            "median_speed_px_s": median([segment[0] for segment in segments]),
            "features": features,
        })

    candidates = []
    for tracks in class_tracks.values():
        if len(tracks) < 12:
            continue
        matrix = np.asarray([track["features"] for track in tracks], dtype=float)
        scaled = RobustScaler().fit_transform(matrix)
        model = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42, n_jobs=1,
        ).fit(scaled)
        labels = model.predict(scaled)
        scores = -model.score_samples(scaled)
        for track, label, score in zip(tracks, labels, scores):
            if label != -1:
                continue
            track.pop("features")
            track["type"] = "isolation_forest_motion_outlier"
            track["anomaly_score"] = float(score)
            candidates.append(track)
    return sorted(candidates, key=lambda item: item["anomaly_score"], reverse=True)

@app.get("/api/videos/{vid}/anomalies")
def anomalies(vid: int):
    c = db.conn(); video(c, vid)
    candidates = motion_anomalies(c, vid)
    model_candidates = model_motion_anomalies(c, vid)
    reviews = {row["anomaly_id"]: dict(row) for row in c.execute(
        "SELECT anomaly_id,status,note,updated FROM anomaly_reviews WHERE video_id=?", (vid,)
    )}
    for candidate in candidates + model_candidates:
        review = reviews.get(candidate["id"])
        candidate["review_status"] = review["status"] if review else "pending"
        candidate["review_note"] = review["note"] if review else None
    return {
        "video_id": vid,
        "method": "Robust within-class high-speed outliers of per-track median image-plane centroid speed (MAD z-score >= 3.5; at least 8 observations and 8 tracks per class).",
        "coordinate_space": "image-plane pixels per second; camera motion and tracking switches can cause outliers",
        "model_configured": True,
        "model_name": "Isolation Forest",
        "model_method": "Deterministic unsupervised per-video, per-class fit on track speed, path efficiency, confidence, and observation-count features; at least 12 tracks per class. Scores are not probabilities.",
        "candidates": candidates,
        "model_candidates": model_candidates,
    }

class AnomalyReview(BaseModel):
    status: str
    note: str | None = None

@app.post("/api/videos/{vid}/anomalies/{anomaly_id}/review")
def review_anomaly(vid: int, anomaly_id: str, review: AnomalyReview):
    if review.status not in {"confirmed", "dismissed"}:
        raise HTTPException(422, "Review status must be 'confirmed' or 'dismissed'.")
    c = db.conn(); video(c, vid)
    valid_ids = {item["id"] for item in motion_anomalies(c, vid)}
    valid_ids.update(item["id"] for item in model_motion_anomalies(c, vid))
    if anomaly_id not in valid_ids:
        raise HTTPException(404, "Unknown anomaly candidate")
    c.execute(
        "INSERT INTO anomaly_reviews(video_id,anomaly_id,status,note) VALUES(?,?,?,?) "
        "ON CONFLICT(video_id,anomaly_id) DO UPDATE SET status=excluded.status,note=excluded.note,updated=CURRENT_TIMESTAMP",
        (vid, anomaly_id, review.status, review.note),
    )
    c.commit()
    return {"video_id": vid, "anomaly_id": anomaly_id, "status": review.status}

@app.get("/api/videos/{vid}/detections")
def detections(vid: int, t: float):
    c = db.conn(); r = video(c, vid)
    if not r["fps"]:
        return []
    nearest = c.execute("SELECT frame,t FROM detections WHERE video_id=? ORDER BY ABS(t-?) LIMIT 1", (vid, t)).fetchone()
    stride = r["analysis_stride"] or pipeline.STRIDE
    if not nearest or abs(nearest["t"] - t) > stride / (2 * r["fps"]) + 0.01:
        return []
    rows = c.execute("SELECT frame,t,track_id,cls,conf,x1,y1,x2,y2 FROM detections WHERE video_id=? AND frame=? ORDER BY track_id", (vid, nearest["frame"])).fetchall()
    return [dict(row) for row in rows]

@app.get("/api/videos/{vid}/frame")
def frame(vid: int, t: float):
    """Evidence frame extracted from the original file at time t (seconds)."""
    cap = cv2.VideoCapture(video(db.conn(), vid)["path"])
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = cap.read(); cap.release()
    if not ok: raise HTTPException(404, "Frame not available")
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")

@app.get("/api/videos/{vid}/evidence")
def evidence(vid: int, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100)):
    c = db.conn(); video(c, vid)
    total = c.execute(
        "SELECT COUNT(*) FROM (SELECT track_id,cls FROM detections WHERE video_id=? GROUP BY track_id,cls)",
        (vid,),
    ).fetchone()[0]
    rows = c.execute(
        "SELECT track_id,cls,COUNT(*) observations,MIN(t) start_t,MAX(t) end_t,AVG(conf) mean_conf "
        "FROM detections WHERE video_id=? GROUP BY track_id,cls "
        "ORDER BY observations DESC,track_id LIMIT ? OFFSET ?",
        (vid, limit, offset),
    ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["id"] = f"track-{item['track_id']}-{re.sub(r'[^a-z0-9]+', '-', item['cls'].lower()).strip('-')}"
        item["kind"] = "track_observation_summary"
        item["source"] = "stored_detection_boxes"
        item["reasoning"] = (
            f"{item['observations']} stored detection boxes were assigned to track {item['track_id']} "
            f"({item['cls']}) between {item['start_t']:.2f}s and {item['end_t']:.2f}s; "
            f"mean detector confidence was {(item['mean_conf'] or 0):.0%}."
        )
        item["frames"] = [
            {"label": "first observation", "t": item["start_t"],
             "url": f"/api/videos/{vid}/frame?t={item['start_t']:.3f}"},
            {"label": "last observation", "t": item["end_t"],
             "url": f"/api/videos/{vid}/frame?t={item['end_t']:.3f}"},
        ]
        items.append(item)
    return {
        "video_id": vid, "total_items": total, "offset": offset, "limit": limit,
        "reasoning": "Items summarize persisted detector boxes grouped by tracker ID and class; timestamps and frame links refer to the original uploaded video.",
        "limitations": [
            "Detector labels and track IDs are model outputs, not verified identities or scene-level events.",
            "Track IDs are local to this video and may change after occlusion or tracking errors.",
            "Coordinates are image pixels; depth, physical distance, intent, and causal relationships are unavailable.",
            "Detector confidence describes the object detection, not the certainty of an event interpretation.",
        ],
        "items": items,
    }

class Q(BaseModel):
    question: str
    video_id: int | None = None

def _requested_word_limit(question):
    match = re.search(r"\b(\d{1,3})\s+words?\b", question.lower())
    return min(180, max(40, int(match.group(1)))) if match else 100

def _limit_answer_words(answer, limit):
    words = answer.split()
    if len(words) <= limit:
        return answer
    shortened = " ".join(words[:limit]).rstrip(",;:")
    boundary = max(shortened.rfind("."), shortened.rfind("!"), shortened.rfind("?"))
    if boundary >= len(shortened) * 0.65:
        return shortened[:boundary + 1]
    return shortened + "…"

def _clock(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"

def _plural_class(cls, count):
    if count == 1:
        return cls
    return {
        "person": "people", "traffic light": "traffic lights",
        "fire hydrant": "fire hydrants", "motorcycle": "motorcycles",
    }.get(cls, cls + "s")

def _video_answer(c, row, question):
    vid, duration = row["id"], float(row["duration"] or 0)
    class_rows = c.execute(
        "SELECT cls,COUNT(DISTINCT track_id) tracks,COUNT(*) observations "
        "FROM detections WHERE video_id=? GROUP BY cls ORDER BY tracks DESC,observations DESC",
        (vid,),
    ).fetchall()
    total_tracks = c.execute(
        "SELECT COUNT(*) FROM (SELECT track_id,cls FROM detections "
        "WHERE video_id=? GROUP BY track_id,cls)",
        (vid,),
    ).fetchone()[0]
    if not class_rows:
        return {"answer": "No tracked objects were detected in this video.", "evidence": []}

    top_classes = class_rows[:5]
    class_text = ", ".join(
        f"{item['tracks']} {_plural_class(item['cls'], item['tracks'])}"
        for item in top_classes
    )
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
    vehicle_tracks = sum(item["tracks"] for item in class_rows if item["cls"] in vehicle_classes)
    people_tracks = sum(item["tracks"] for item in class_rows if item["cls"] == "person")

    window = max(duration / 3, 1)
    activity = []
    for index in range(3):
        start, end = index * window, min((index + 1) * window, duration)
        if index == 2:
            count = c.execute(
                "SELECT COUNT(*) FROM detections WHERE video_id=? AND t>=? AND t<=?",
                (vid, start, end),
            ).fetchone()[0]
        else:
            count = c.execute(
                "SELECT COUNT(*) FROM detections WHERE video_id=? AND t>=? AND t<?",
                (vid, start, end),
            ).fetchone()[0]
        activity.append((count, start, end))
    busiest_count, busiest_start, busiest_end = max(activity)

    candidates = [
        item for item in motion_anomalies(c, vid)
        if item["cls"] in vehicle_classes and item["observations"] >= 12
    ][:3]
    if candidates:
        candidate_text = "; ".join(
            f"{item['cls']} track #{item['track_id']} at {_clock(item['start_t'])}"
            for item in candidates
        )
        motion_sentence = (
            f"Vehicle-motion outlier flags appear near {candidate_text}. "
            "They measure image-plane changes and may reflect perspective or tracking switches."
        )
    else:
        motion_sentence = "No qualifying vehicle-motion outlier was found in the stored tracks."

    if vehicle_tracks and people_tracks:
        scene = "The detector labels suggest a road-traffic scene with vehicles and pedestrians."
    elif vehicle_tracks:
        scene = "The detector labels suggest a vehicle-traffic scene."
    else:
        scene = "The detector labels show " + class_text + "."

    accident_question = bool(re.search(
        r"\b(?:acci\w*|crash\w*|collision\w*|impact|hit|injur\w*|near miss)\b",
        question.lower(),
    ))
    if accident_question:
        if candidates:
            flagged_times = "; ".join(
                f"{item['cls']} track #{item['track_id']} near {_clock(item['start_t'])}"
                for item in candidates
            )
        else:
            flagged_times = "no qualifying vehicle-motion outlier was found"
        answer = (
            f"{scene} Across {_clock(duration)} of footage, the tracker recorded "
            f"{total_tracks} object tracks, including {class_text}. "
            f"The busiest interval was {_clock(busiest_start)}–{_clock(busiest_end)}. "
            f"Brief vehicle-motion outlier flags appear near {flagged_times}. "
            "They are image-plane review clues, not collision detections; perspective or tracking "
            "switches may cause them. Stored tracks do not confirm impact, injury, or cause, "
            "so an accident cannot be verified from these observations alone."
        )
    else:
        answer = (
            f"{scene} Across {_clock(duration)} of footage, the tracker followed "
            f"{total_tracks} object tracks, including {class_text}. "
            f"The busiest interval was {_clock(busiest_start)}–{_clock(busiest_end)}, "
            f"with {busiest_count} tracked observations. {motion_sentence} "
            "This is a summary of detector labels and image-plane tracks, not verified actions, "
            "physical speeds, or causes."
        )

    evidence = []
    primary = c.execute(
        "SELECT track_id,MIN(t) start_t,MAX(t) end_t FROM detections "
        "WHERE video_id=? GROUP BY track_id ORDER BY COUNT(*) DESC LIMIT 1",
        (vid,),
    ).fetchone()
    if primary:
        evidence.append({
            "video_id": vid, "track": primary["track_id"],
            "start": primary["start_t"], "end": primary["end_t"],
        })
    for item in candidates:
        if not any(entry["track"] == item["track_id"] for entry in evidence):
            evidence.append({
                "video_id": vid, "track": item["track_id"],
                "start": item["start_t"], "end": item["end_t"],
            })
    return {
        "answer": _limit_answer_words(answer, _requested_word_limit(question)),
        "evidence": evidence,
    }

@app.post("/api/chat")
def chat(q: Q):
    """Summarize stored tracks and report motion candidates without treating them as verified events."""
    c = db.conn()
    try:
        r = video(c, q.video_id) if q.video_id else c.execute(
            "SELECT * FROM videos WHERE status='done' ORDER BY id DESC"
        ).fetchone()
        if not r or r["status"] != "done":
            return {"answer": "No processed video is available to summarize.", "evidence": []}
        question = q.question.strip()
        m = re.search(r"(?:object|track|id)\s*#?(\d+)", question.lower())
        if m:
            tid = int(m.group(1))
            x = c.execute(
                "SELECT cls,COUNT(*) n,MIN(t) a,MAX(t) b FROM detections "
                "WHERE video_id=? AND track_id=?",
                (r["id"], tid),
            ).fetchone()
            if not x["n"]:
                return {"answer": f"INSUFFICIENT EVIDENCE: track {tid} was not observed.", "evidence": []}
            return {
                "answer": f"Track {tid} ({x['cls']}) was observed in {x['n']} frames from "
                          f"{_clock(x['a'])} to {_clock(x['b'])}. Movement history is available "
                          f"at /api/videos/{r['id']}/tracks/{tid}.",
                "evidence": [{"video_id": r["id"], "start": x["a"], "end": x["b"], "track": tid}],
            }
        return _video_answer(c, r, question)
    finally:
        c.close()

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="ui")
