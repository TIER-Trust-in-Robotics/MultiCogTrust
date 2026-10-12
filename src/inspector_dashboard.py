"""Small localhost inspector; camera frames stay in memory and are never logged."""

from __future__ import annotations
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit


class InspectorDashboard:
    def __init__(self, port=8765):
        self.lock = threading.Lock()
        self.state = {"status": "loading", "source": "initializing", "history": []}
        self.history = deque(maxlen=20)
        self.jpeg = None
        self.stop_requested = threading.Event()
        self.on_stop = None
        token = secrets.token_urlsafe(18)
        dashboard = self
        html = (
            Path(__file__)
            .with_name("inspector.html")
            .read_text()
            .replace("SESSION_TOKEN", token)
        )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body, mime, code=200):
                self.send_response(code)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    self.reply(html.encode(), "text/html; charset=utf-8")
                elif path == "/state":
                    with dashboard.lock:
                        body = json.dumps(
                            {
                                **dashboard.state,
                                "history": list(dashboard.history),
                                "server_now": time.monotonic(),
                            },
                            allow_nan=False,
                        ).encode()
                    self.reply(body, "application/json")
                elif path == "/frame.jpg":
                    with dashboard.lock:
                        frame = dashboard.jpeg
                    self.reply(frame or b"", "image/jpeg", 200 if frame else 204)
                else:
                    self.reply(b"Not found", "text/plain", 404)

            def do_POST(self):
                if (
                    urlsplit(self.path).path != "/stop"
                    or self.headers.get("X-Session") != token
                ):
                    self.reply(b"Forbidden", "text/plain", 403)
                    return
                dashboard.stop_requested.set()
                if dashboard.on_stop:
                    dashboard.on_stop()
                self.reply(b"{}", "application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def set_stop_callback(self, callback):
        self.on_stop = callback
        if callback and self.stop_requested.is_set():
            callback()

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def vision(self, summary, jpeg):
        with self.lock:
            self.state["vision"] = summary
            if jpeg:
                self.jpeg = jpeg

    def record(self, event):
        with self.lock:
            if event.get("event") == "prediction" or "emotion" in event:
                self.state["prediction"] = event
                self.history.appendleft(
                    {
                        k: event.get(k)
                        for k in ("start_sec", "end_sec", "text", "status", "emotion")
                    }
                )
            elif event.get("event") == "stream_summary":
                self.state["metrics"] = event
            else:
                self.state["last_event"] = event

    def hold(self, seconds):
        self.stop_requested.wait(seconds)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
