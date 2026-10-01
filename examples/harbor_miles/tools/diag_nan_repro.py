"""Reproduce: does SGLang return None/NaN output logprobs when a request's prompt nearly fills the context window
and no max_tokens is sent (engine clamps max_new_tokens to the remaining window)? Run inside the Miles SIF."""
import json, math, os, subprocess, sys, time, urllib.request
MODEL = sys.argv[1]; CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
port = 31555
srv = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL, "--context-length", str(CTX),
                        "--port", str(port), "--host", "127.0.0.1", "--mem-fraction-static", "0.7", "--disable-cuda-graph"],
                       stdout=open("/tmp/sglang_repro.log", "w"), stderr=subprocess.STDOUT)
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:300]}
for _ in range(240):
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5); break
    except Exception: time.sleep(5)
else:
    print("server did not start"); print(open("/tmp/sglang_repro.log").read()[-3000:]); sys.exit(1)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
filler = " ".join(["alpha"] * 20000)
def run(room, use_ids):
    # build input_ids of length CTX - room
    ids = tok(filler)["input_ids"][: CTX - room]
    body = {"input_ids": ids, "sampling_params": {"temperature": 1.0, "top_p": 1.0}, "return_logprob": True}
    r = post("/generate", body)
    if "http_error" in r: return f"room={room}: HTTP {r['http_error']} {r['body'][:160]}"
    mi = r.get("meta_info", {}); otl = mi.get("output_token_logprobs") or []
    lps = [t[0] for t in otl]
    bad = [(i, v) for i, v in enumerate(lps) if v is None or (isinstance(v, float) and not math.isfinite(v))]
    return f"room={room}: finish={mi.get('finish_reason')} n_out={len(otl)} bad={bad[:5]} last3={lps[-3:]}"
for room in (64, 8, 4, 2, 1, 0):
    print(run(room, True), flush=True)
srv.terminate()
