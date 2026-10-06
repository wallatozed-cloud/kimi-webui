#!/usr/bin/env python3
"""Kimi K3 Premium — streaming NN chat (NVIDIA NIM) with voice + image studio.
Python stdlib only.
  - /api/chat/stream : SSE streaming chat
  - /api/image/gen  : Flux.1-schnell text->image (NVIDIA)
  - /api/image/edit : Flux.1-schnell img2img prompt-guided edit (NVIDIA)
"""

import http.server
import json
import os
import sys
import urllib.request
import urllib.error
import base64
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

PORT = int(os.getenv("PORT", os.getenv("KIMI_WEBUI_PORT", "8788")))
PASSWORD = os.getenv("KIMI_WEBUI_PASSWORD", "")
API_KEY = os.getenv("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = os.getenv("KIMI_MODEL", "deepseek-ai/deepseek-v4.1-flash")
FLUX = "black-forest-labs/flux.1-schnell"

if not PASSWORD:
    print("[!!] KIMI_WEBUI_PASSWORD not set in .env", file=sys.stderr)
    sys.exit(1)

with open(os.path.join(ROOT, "index.html"), encoding="utf-8") as f:
    HTML = f.read()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, data, status=200):
        b = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def _html(self, h):
        b = h.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def _require_auth(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] == PASSWORD:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Bearer realm="kimi"')
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        return False

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n)

    def do_GET(self):
        if self.path == "/":
            self._html(HTML)
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/auth":
            data = json.loads(self._read_body())
            if data.get("password") == PASSWORD:
                self._json({"token": PASSWORD, "model": data.get("model", MODEL)})
            else:
                self._json({"error": "invalid"}, 401)
            return

        if self.path == "/api/chat/stream":
            if not self._require_auth():
                return
            data = json.loads(self._read_body())
            messages = data.get("messages", [])
            model = data.get("model") or MODEL
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            req = urllib.request.Request(
                BASE_URL + "/chat/completions",
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "temperature": 0.6,
                    "max_tokens": 4096,
                    "stream": True,
                }).encode(),
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    while True:
                        raw = resp.readline()
                        if not raw:
                            break
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data: "):
                            continue
                        chunk_data = line[6:]
                        if chunk_data == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk_data)
                            delta = obj["choices"][0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                self.wfile.write(f"data: {json.dumps({'t': text})}\n\n".encode())
                                self.wfile.flush()
                        except Exception:
                            continue
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:400]
                try:
                    self.wfile.write(f"data: {json.dumps({'err': f'API {e.code}: {body}'})}\n\n".encode())
                except Exception:
                    pass
            finally:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            return

        if self.path == "/api/image/gen":
            if not self._require_auth():
                return
            data = json.loads(self._read_body())
            prompt = data.get("prompt", "").strip()
            if not prompt:
                self._json({"error": "prompt required"}, 400)
                return
            body = {
                "prompt": prompt,
                "mode": "base",
                "cfg_scale": 5,
                "width": 1024,
                "height": 1024,
                "sample_count": 1,
                "seed": data.get("seed", 0),
                "steps": 4,
            }
            try:
                b64 = self._flux_call(body)
                self._json({"image": "data:image/jpeg;base64," + b64})
            except Exception as e:
                self._json({"error": str(e)}, 502)
            return

        if self.path == "/api/image/edit":
            if not self._require_auth():
                return
            data = json.loads(self._read_body())
            prompt = data.get("prompt", "").strip()
            image_b64 = data.get("image", "")
            if not prompt or not image_b64:
                self._json({"error": "image + prompt required"}, 400)
                return
            # Strip "data:image/...;base64," prefix
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            body = {
                "image": image_b64,
                "prompt": prompt,
                "mode": "img2img",
                "cfg_scale": 5,
                "width": 1024,
                "height": 1024,
                "sample_count": 1,
                "seed": data.get("seed", 0),
                "steps": 4,
            }
            try:
                b64 = self._flux_call(body)
                self._json({"image": "data:image/jpeg;base64," + b64})
            except Exception as e:
                self._json({"error": str(e)}, 502)
            return

        self.send_error(404)

    def _flux_call(self, body):
        req = urllib.request.Request(
            f"https://ai.api.nvidia.com/v1/genai/{FLUX}",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.load(resp.read())
            arts = d.get("artifacts", [])
            if not arts or "base64" not in arts[0]:
                raise RuntimeError(f"no image in response: {json.dumps(d)[:200]}")
            return arts[0]["base64"]


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32


def main():
    # Bind to 0.0.0.0 for containerized environments (Render), 127.0.0.1 otherwise
    bind_host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    server = QuietServer((bind_host, PORT), Handler)
    print(f"[kimi-webui] http://127.0.0.1:{PORT}")
    print(f"[kimi-webui] stream: {MODEL}")
    print(f"[kimi-webui] image: {FLUX}")
    server.serve_forever()


if __name__ == "__main__":
    main()
