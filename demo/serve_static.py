#!/usr/bin/env python3
"""Local server for the static site and the in-browser engine, with the same routes and headers as Vercel.

  /            -> demo/static/index.html       /studio -> demo/static/studio.html
  /static/...  -> demo/static/...              anything else -> demo/static/...
  /models/...  -> results/live_onnx/...        (engine/config.js MODEL_BASE)
  /hf/<org>/<name>/resolve/<revision>/<path>   -> the local Hugging Face cache (engine/config.js HF_BASE)
  /health      -> 404, as on Vercel (the studio reads 404 as "static host: use the in-browser engine")

Headers are read from demo/static/vercel.json, so local and deployed pages cannot drift. Standard library only.
Local use only: requests must name Host 127.0.0.1, localhost or [::1] on the bound port (blocks DNS rebinding), and a
path with an encoded separator (%2f, %5c), a '..' segment or a leftover escape (double encoding) is refused, so the
header rules and the file lookup always see the same decoded, normalised path.
Dev extras: --mount PREFIX=DIR serves another directory (e.g. H.264 test clips for a stand-in camera), and
--upload DIR accepts POST /__upload/<name> (e.g. to save the frames a browser run captured). An upload must carry the
header `X-Dev-Upload: 1` (a cross-site page cannot send it without a CORS preflight, which this server never answers)
and a Content-Length of at most 64 MiB.
"""
import argparse
import http.server
import json
import os
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / 'demo' / 'static'
# Vercel header rules; their sources are plain regexes ('/(.*)', '/((?!static/engine/).*)'), and every matching rule applies.
RULES = [(re.compile(r['source']), r['headers']) for r in json.loads((STATIC / 'vercel.json').read_text())['headers']]
HF_HUB = Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub')
LOCAL_HOSTS = ('127.0.0.1', 'localhost', '[::1]')
UPLOAD_HEADER = 'X-Dev-Upload'
UPLOAD_MAX = 64 << 20        # bytes; the largest G6 upload (one captured utterance) was 8.8 MB


def clean_path(raw):
    """The decoded, normalised URL path (trailing slash kept), or None if it must be refused."""
    path = raw.split('?', 1)[0].split('#', 1)[0]
    if re.search(r'%(2f|5c)', path, re.I): return None
    path = unquote(path)
    if any(c in path for c in '%\\\0') or '..' in path.split('/'): return None
    norm = posixpath.normpath('/' + path.lstrip('/'))
    return norm + '/' if path.endswith('/') and norm != '/' else norm


def make_handler(mounts, upload):
    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.js': 'text/javascript',
                          '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'}

        def parse_request(self):
            self.url_path = None                # error replies before the checks below get the headers of '/'
            if not super().parse_request(): return False
            port = self.server.server_address[1]
            hosts = {f'{h}:{port}' for h in LOCAL_HOSTS} | (set(LOCAL_HOSTS) if port == 80 else set())
            if (self.headers.get('Host') or '').lower() not in hosts:
                self.send_error(403, 'Local use only: Host must be 127.0.0.1, localhost or [::1] on this port'); return False
            path = clean_path(self.path)
            if path is None:
                self.send_error(400, "Encoded separator, '..' segment or double encoding in the path"); return False
            self.url_path = path
            if path.rstrip('/') == '/health': self.send_error(404); return False
            return True

        def translate_path(self, _raw):         # self.url_path: decoded and normalised once, in parse_request
            path = self.url_path
            if path == '/': path = '/index.html'
            elif path.rstrip('/') == '/studio': path = '/studio.html'
            if path.startswith('/hf/'):         # /hf/org/name/resolve/rev/file -> models--org--name/snapshots/rev/file
                parts = path[4:].split('/')
                if len(parts) >= 5 and parts[2] == 'resolve':
                    self.directory = str(HF_HUB / f'models--{parts[0]}--{parts[1]}' / 'snapshots')
                    return super().translate_path('/' + '/'.join(parts[3:]))
            for prefix, root in mounts:
                if path.startswith(prefix + '/'):
                    self.directory = str(root)
                    return super().translate_path(path[len(prefix):])
            self.directory = str(STATIC)
            return super().translate_path(path[len('/static'):] if path.startswith('/static/') else path)

        def end_headers(self):
            for pattern, headers in RULES:
                if pattern.fullmatch(self.url_path or '/'):
                    for h in headers: self.send_header(h['key'], h['value'])
            self.send_header('Cache-Control', 'no-cache')
            super().end_headers()

        def do_POST(self):
            name = posixpath.basename(self.url_path)
            if not upload or self.url_path != '/__upload/' + name or not name or name.startswith('.'):
                self.send_error(404); return
            if UPLOAD_HEADER not in self.headers:
                self.send_error(403, f'Uploads need the {UPLOAD_HEADER} header'); return
            size = self.headers.get('Content-Length')
            if size is None: self.send_error(411); return
            if not (size.isascii() and size.isdigit()): self.send_error(400, 'Bad Content-Length'); return
            if int(size) > UPLOAD_MAX: self.send_error(413, f'Uploads are capped at {UPLOAD_MAX} bytes'); return
            (upload / name).write_bytes(self.rfile.read(int(size)))
            self.send_response(204); self.end_headers()

        def log_message(self, fmt, *args): pass   # quiet: frames and model files are many requests

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8123)
    parser.add_argument('--models', default=str(ROOT / 'results' / 'live_onnx'), help='directory served at /models')
    parser.add_argument('--mount', action='append', default=[], metavar='PREFIX=DIR', help='serve DIR at /PREFIX (dev only)')
    parser.add_argument('--upload', metavar='DIR', help='accept POST /__upload/<name> into DIR (dev only)')
    parser.add_argument('--self_test', action='store_true', help='check the path rules and exit')
    args = parser.parse_args()
    if args.self_test:
        for raw, want in [('/', '/'), ('/studio/?x=1', '/studio/'), ('//static/./engine//a.js', '/static/engine/a.js'),
                          ('/%73tatic/engine/a.js', '/static/engine/a.js'), ('/a%20b.mp4', '/a b.mp4'),
                          ('/static/engine/..%2fstudio.html', None), ('/static/engine/..%2Fx', None), ('/a%5cb', None),
                          ('/static/engine/../studio.html', None), ('/static/%2e%2e/x', None), ('/x/..', None),
                          ('/..%252fx', None), ('/a\\b', None), ('/a%00.js', None)]:
            assert clean_path(raw) == want, (raw, clean_path(raw), want)
        print('serve_static path rules: ok'); return
    mounts = [('/models', Path(args.models))] + [('/' + p.strip('/'), Path(d)) for p, d in (m.split('=', 1) for m in args.mount)]
    upload = Path(args.upload) if args.upload else None
    if upload: upload.mkdir(parents=True, exist_ok=True)
    class Server(http.server.ThreadingHTTPServer):
        request_queue_size = 64   # the default 5 drops worker imports under load (measured: 42/80 starts failed)
    server = Server((args.host, args.port), make_handler(mounts, upload))
    print(f'http://{args.host}:{args.port}/studio  (models from {args.models})')
    server.serve_forever()


if __name__ == '__main__':
    main()
