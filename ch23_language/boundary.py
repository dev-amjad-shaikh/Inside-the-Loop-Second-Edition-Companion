"""Cost of crossing a language or process boundary once per step (Chapter 23).

Python is the caller in every case. The callee receives a JSON payload (a tool
call or a transcript slice), parses it and returns a small JSON acknowledgement.

  in-process  a Python function (json.loads + json.dumps), no boundary
  binding     a Rust function compiled as a CPython extension with PyO3 (optional;
              built by the driver if cargo can fetch pyo3)
  stdio       JSON lines to a long-lived child process over pipes (the shape of
              `codex app-server --listen stdio://`)
  http-naive  HTTP/1.1 POST with keep-alive to a local server (the shape of an
              HTTP/SSE runtime such as Rusty Server), using http.client and
              http.server from the standard library with default socket options
  http        the same with TCP_NODELAY on both ends: the server writes headers and
              body in separate send() calls, and Nagle's algorithm plus delayed ACK
              then holds the body for about 40 ms

Usage: python3 boundary.py --out result.json [--binding-dir DIR]
"""
import argparse
import http.client
import http.server
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time

perf = time.perf_counter

CHILD = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    sys.stdout.write(json.dumps({"ok": True, "n": len(msg["payload"])}) + "\n")
    sys.stdout.flush()
'''


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        msg = json.loads(body)
        out = json.dumps({"ok": True, "n": len(msg["payload"])}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class NoDelayHandler(Handler):
    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def in_process(text):
    msg = json.loads(text)
    return json.dumps({"ok": True, "n": len(msg["payload"])})


def timeit(fn, reps, rounds=3):
    """Three rounds of `reps` calls; report the middle round's median and p99 plus the spread."""
    for _ in range(50):
        fn()
    meds, p99s = [], []
    for _ in range(rounds):
        out = []
        for _ in range(reps):
            t0 = perf()
            fn()
            out.append((perf() - t0) * 1e6)
        out.sort()
        meds.append(statistics.median(out))
        p99s.append(out[int(0.99 * (len(out) - 1))])
    return {"median_us": round(statistics.median(meds), 2), "p99_us": round(statistics.median(p99s), 2),
            "median_us_min": round(min(meds), 2), "median_us_max": round(max(meds), 2),
            "reps": reps, "rounds": rounds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--binding-dir")
    p.add_argument("--reps", type=int, default=2000)
    a = p.parse_args()
    payloads = {"1KB": "x" * 1000, "50KB": "y" * 50_000}
    res = {"python": sys.version.split()[0], "cases": {}}

    binding = None
    if a.binding_dir:
        sys.path.insert(0, a.binding_dir)
        try:
            import ch23_binding
            binding = ch23_binding.ack
        except ImportError as e:
            res["binding_error"] = str(e)

    child = subprocess.Popen([sys.executable, "-c", CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    conn_naive = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    srv2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NoDelayHandler)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    conn = http.client.HTTPConnection("127.0.0.1", srv2.server_address[1])
    conn.connect()
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    for label, payload in payloads.items():
        text = json.dumps({"id": "call_step", "payload": payload})
        body = text.encode()

        def stdio():
            child.stdin.write(text + "\n")
            child.stdin.flush()
            return child.stdout.readline()

        def http_naive():
            conn_naive.request("POST", "/step", body=body, headers={"Content-Type": "application/json"})
            return conn_naive.getresponse().read()

        def http_post():
            conn.request("POST", "/step", body=body, headers={"Content-Type": "application/json"})
            return conn.getresponse().read()

        res["cases"][f"in-process|{label}"] = timeit(lambda: in_process(text), a.reps)
        if binding:
            res["cases"][f"binding|{label}"] = timeit(lambda: binding(text), a.reps)
        res["cases"][f"stdio|{label}"] = timeit(stdio, a.reps)
        res["cases"][f"http-naive|{label}"] = timeit(http_naive, 200)
        res["cases"][f"http|{label}"] = timeit(http_post, a.reps)
    child.stdin.close()
    child.wait()
    srv.shutdown()
    srv2.shutdown()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
