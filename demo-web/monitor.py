"""Event recording for the inline V2 notebook dashboard."""
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
from urllib.parse import urlsplit
import json
import os
import sqlite3
import subprocess
import time
import uuid

import cv2
import numpy as np

WEB = Path(__file__).resolve().parent
STORE = WEB / "recordings"
PRE, QUIET, FPS = 3, 1, 16
LIVE_QUIET = 3
ALERT = {"fight", "suspected"}


@contextmanager
def connect():
    db = sqlite3.connect(STORE / "events.sqlite", timeout=15)
    try:
        with db:
            yield db
    finally:
        db.close()


def save_event(key, source, start, end, score, kind):
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?)",
                   (key, source, start, end, score, kind, datetime.now().isoformat(timespec="seconds"),
                    "pending", key + ".mp4"))


def encode(source, target, start, duration):
    tmp = target.with_name(target.stem + "." + uuid.uuid4().hex + ".mp4")
    try:
        subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-ss",str(start),"-i",str(source),
                        "-t",str(duration),"-map","0:v:0","-an","-c:v","libx264","-pix_fmt","yuv420p",
                        "-movflags","+faststart",str(tmp)],check=True)
        os.replace(tmp,target)
    finally:
        tmp.unlink(missing_ok=True)


def intervals(rows):
    group = None
    for row in rows.sort_values("start_s").itertuples():
        if row.decision not in ALERT:
            continue
        if group is None or row.start_s > group[1] + QUIET:
            if group:
                yield group
            group = [row.start_s,row.end_s,row.p_action,row.decision]
        else:
            group[1] = max(group[1],row.end_s)
            group[2] = max(group[2],row.p_action)
            if row.decision == "fight":
                group[3] = "fight"
    if group:
        yield group


def prepare(windows):
    STORE.mkdir(exist_ok=True)
    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, source TEXT, start REAL, end REAL, score REAL, kind TEXT, created TEXT, state TEXT, file TEXT)")
    sources = {}
    for name, rows in windows.groupby("path"):
        path = Path(name)
        sources[str(len(sources))] = path
        cap = cv2.VideoCapture(str(path))
        try:
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        for onset, last, score, kind in intervals(rows):
            start, end = max(0,onset-PRE), min(duration,last+QUIET)
            ident = sha256(f"{path.resolve()}|{path.stat().st_mtime_ns}|{start}|{end}|{score}|{kind}".encode()).hexdigest()[:24]
            with connect() as db:
                if db.execute("SELECT 1 FROM events WHERE id=? AND state='deleted'", (ident,)).fetchone():
                    continue
            target = STORE / (ident + ".mp4")
            if not target.exists():
                encode(path,target,start,end-start)
            save_event(ident,path.name,start,end,score,kind)
    return sources


class Camera:
    def __init__(self, predict):
        self.predict = predict
        self.lock = Lock()
        self.worker = None
        self.incoming = Queue(maxsize=8)
        self.stop = Event()
        self.status = "Camera đang tắt"
        self.frame = None

    def start(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                return
            self.incoming = Queue(maxsize=8)
            self.stop.clear()
            self.status = "Đang mở camera"
            self.worker = Thread(target=self.run,daemon=True)
            self.worker.start()

    def capture(self, queue, finished):
        try:
            origin = None
            count = 0
            last = time.monotonic()
            while not self.stop.is_set():
                try:
                    stamp,frame = self.incoming.get(timeout=.3)
                except Empty:
                    if time.monotonic()-last > 6:
                        raise RuntimeError("Không nhận được hình từ trình duyệt")
                    continue
                last = stamp
                if origin is None: origin = stamp
                frame = cv2.resize(frame,(640,360))
                self.frame = cv2.imencode(".jpg",frame)[1].tobytes()
                due = int((stamp-origin)*FPS)
                while count <= due:
                    queue.put(frame,timeout=1)
                    count += 1
        except Exception as exc:
            self.status = "Dừng camera: " + (str(exc) or "suy luận không theo kịp luồng")
            self.stop.set()
        finally:
            finished.set()

    def run(self):
        queue, finished = Queue(maxsize=FPS*8), Event()
        producer = Thread(target=self.capture,args=(queue,finished),daemon=True)
        producer.start()
        ring = deque(maxlen=(PRE+1)*FPS)
        writer = None
        count = quiet = 0
        def close_clip():
            nonlocal writer
            if writer is None:
                return
            writer.release()
            writer = None
            encode(raw, STORE/(key+".mp4"), 0, (count-start)/FPS)
            save_event(key,"Webcam 0",start/FPS,count/FPS,peak,kind)
            raw.unlink()
        try:
            while not finished.is_set() or not queue.empty():
                try:
                    frame = queue.get(timeout=.2)
                except Empty:
                    continue
                ring.append(frame)
                count += 1
                if writer:
                    writer.write(frame)
                if count % (FPS//2):
                    continue
                decision, score = self.predict(list(ring)[-FPS:])
                if not self.stop.is_set():
                    self.status = f"{decision} · {score:.1%}"
                if decision in ALERT:
                    quiet = 0
                    if writer is None:
                        key = uuid.uuid4().hex
                        raw = STORE/(key+".avi")
                        writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"MJPG"),FPS,(640,360))
                        if not writer.isOpened():
                            writer = None
                            raise RuntimeError("Không ghi được đoạn camera")
                        start,peak,kind = count-len(ring),score,decision
                        for previous in ring:
                            writer.write(previous)
                    peak = max(peak,score)
                    if decision == "fight":
                        kind = decision
                elif writer:
                    quiet += FPS//2
                    if quiet >= LIVE_QUIET*FPS:
                        close_clip()
            close_clip()
            if not self.status.startswith("Dừng camera:"):
                self.status = "Camera đang tắt"
        except Exception as exc:
            self.status = f"Lỗi ghi camera: {exc}"
        finally:
            self.stop.set()
            if writer:
                writer.release()
            producer.join(timeout=2)
            self.frame = None


def launch(windows, predict, port=0):
    sources = prepare(windows)
    silent_sources = {}
    for key,path in sources.items():
        cache = STORE / "silent"
        cache.mkdir(exist_ok=True)
        target = cache / f"{path.stem}-{path.stat().st_mtime_ns}.mp4"
        if not target.exists():
            tmp = target.with_suffix(".tmp.mp4")
            subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-i",str(path),"-map","0:v:0","-c:v","copy","-an","-movflags","+faststart",str(tmp)],check=True)
            os.replace(tmp,target)
        silent_sources[key] = target
    camera = Camera(predict)
    timeline = {key: windows.loc[windows.path.eq(str(path)), ["start_s","end_s","p_action","decision"]].to_dict("records")
                for key,path in sources.items()}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send_data(self, data, mime):
            self.send_response(200)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route in ("/", "/index.html"):
                return self.send_data((WEB/"index.html").read_bytes(),"text/html; charset=utf-8")
            if route == "/timeline":
                return self.send_data(json.dumps(timeline).encode(),"application/json")
            if route == "/api":
                with connect() as db:
                    db.row_factory = sqlite3.Row
                    events = [dict(row) for row in db.execute("SELECT * FROM events WHERE state!='deleted' ORDER BY created DESC,id")]
                state = dict(events=events,sources={k:p.name for k,p in sources.items()},status=camera.status,
                             live=bool(camera.worker and camera.worker.is_alive()),preview=camera.frame is not None)
                return self.send_data(json.dumps(state).encode(),"application/json")
            if route == "/preview" and camera.frame:
                return self.send_data(camera.frame,"image/jpeg")
            prefix, _, key = route.lstrip("/").partition("/")
            path = silent_sources.get(key) if prefix=="source" else None
            if prefix=="clip":
                with connect() as db:
                    row = db.execute("SELECT file FROM events WHERE id=? AND state!='deleted'",(key,)).fetchone()
                path = STORE/row[0] if row else None
            if not path or not path.is_file():
                return self.send_error(404)
            size = path.stat().st_size
            start, end = 0, size-1
            requested = self.headers.get("Range")
            if requested:
                try:
                    unit, limits = requested.split("=")
                    left, right = limits.split("-")
                    if unit != "bytes" or not (left or right):
                        raise ValueError
                    start = int(left) if left else max(0,size-int(right))
                    end = min(int(right),end) if left and right else end
                    if not 0 <= start <= end:
                        raise ValueError
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range",f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type","video/mp4")
            self.send_header("Accept-Ranges","bytes")
            if requested:
                self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length",str(end-start+1))
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    remaining = end-start+1
                    while remaining:
                        block = f.read(min(remaining,65536))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (ConnectionError,OSError):
                pass

        def do_POST(self):
            local_origin = f"http://127.0.0.1:{self.server.server_port}"
            public_origin = os.environ.get("SENTINEL_PUBLIC_ORIGIN", "").rstrip("/")
            allowed_origins = {local_origin}
            if public_origin.startswith("https://"):
                allowed_origins.add(public_origin)
            if self.headers.get("Origin") not in allowed_origins:
                return self.send_error(403)
            if self.path=="/frame":
                if camera.stop.is_set() or not camera.worker or not camera.worker.is_alive():
                    return self.send_error(409)
                size = int(self.headers.get("Content-Length",0))
                if not 0<size<=500000: return self.send_error(413)
                frame = cv2.imdecode(np.frombuffer(self.rfile.read(size),np.uint8),cv2.IMREAD_COLOR)
                if frame is None: return self.send_error(400)
                try:
                    camera.incoming.put_nowait((time.monotonic(),frame))
                except Full:
                    return self.send_error(429)
            elif self.path=="/start":
                camera.start()
            elif self.path=="/stop":
                camera.stop.set()
            else:
                parts=self.path.strip("/").split("/")
                if len(parts)!=2 or parts[0] not in {"saved","discarded"}:
                    return self.send_error(404)
                with connect() as db:
                    state, key = parts
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute("SELECT file,state FROM events WHERE id=?", (key,)).fetchone()
                    if not row or (row[1]=="deleted" and state=="saved"):
                        return self.send_error(404)
                    if state=="discarded":
                        if row[0] != key+".mp4" or not key.isalnum():
                            return self.send_error(409)
                        try:
                            (STORE/row[0]).unlink(missing_ok=True)
                        except OSError:
                            return self.send_error(409,"Clip is busy; retry")
                        state="deleted"
                    db.execute("UPDATE events SET state=? WHERE id=?", (state,key))
            self.send_data(b"{}","application/json")
    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False
    server = LocalServer(("127.0.0.1",port),Handler)
    Thread(target=server.serve_forever,daemon=True).start()
    return server,camera

