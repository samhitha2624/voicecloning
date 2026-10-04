"""Voice cloning web app: upload/record samples, build a voice, synthesize speech.

Uses Coqui XTTS-v2 locally. All samples from a voice are combined into one
speaker conditioning, so more (clean) samples -> a more faithful clone.
"""
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import soundfile as sf
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

BASE = Path(__file__).parent
VOICES_DIR = BASE / "voices"
OUTPUT_DIR = BASE / "outputs"
VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
             "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
MAX_TEXT = 2000

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load XTTS-v2 once (first call downloads ~1.8 GB)."""
    global _model
    with _model_lock:
        if _model is None:
            from TTS.api import TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading XTTS-v2 on {device}...")
            _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        return _model.synthesizer.tts_model


def client_id():
    """Per-browser id; voices are only visible to the browser that made them."""
    cid = request.headers.get("X-Client-Id", "")
    if not (8 <= len(cid) <= 64 and cid.replace("-", "").isalnum()):
        abort(400, "missing client id")
    return cid


def voice_dir(voice_id):
    if not voice_id.isalnum():
        abort(400, "bad voice id")
    d = VOICES_DIR / voice_id
    if not d.is_dir() or read_meta(d).get("owner") != client_id():
        abort(404, "voice not found")
    return d


def read_meta(d):
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def write_meta(d, meta):
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_samples(d, files):
    """Save uploaded WAVs (already converted to mono WAV in the browser)."""
    samples = d / "samples"
    samples.mkdir(exist_ok=True)
    added = 0
    for f in files:
        dest = samples / f"{uuid.uuid4().hex[:10]}.wav"
        f.save(dest)
        try:
            info = sf.info(dest)
        except Exception:
            dest.unlink(missing_ok=True)
            continue
        if info.duration < 1.0:
            dest.unlink(missing_ok=True)
            continue
        added += 1
    # Samples changed, so cached speaker latents are stale.
    (d / "latents.pt").unlink(missing_ok=True)
    return added


def sample_stats(d):
    paths = sorted((d / "samples").glob("*.wav"))
    total = sum(sf.info(p).duration for p in paths)
    return paths, total


def voice_summary(d):
    meta = read_meta(d)
    paths, total = sample_stats(d)
    meta.pop("owner", None)
    return {**meta, "samples": len(paths), "seconds": round(total, 1)}


def get_latents(d):
    """Compute (and cache) speaker conditioning from ALL samples of a voice."""
    cache = d / "latents.pt"
    if cache.exists():
        data = torch.load(cache)
        return data["gpt"], data["spk"]
    model = get_model()
    paths, _ = sample_stats(d)
    if not paths:
        abort(400, "voice has no samples")
    gpt, spk = model.get_conditioning_latents(
        audio_path=[str(p) for p in paths],
        gpt_cond_len=30,
        max_ref_length=60,
        sound_norm_refs=True,
    )
    torch.save({"gpt": gpt.cpu(), "spk": spk.cpu()}, cache)
    return gpt, spk


@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.get("/api/voices")
def list_voices():
    cid = client_id()
    voices = [voice_summary(d) for d in VOICES_DIR.iterdir()
              if (d / "meta.json").exists() and read_meta(d).get("owner") == cid]
    voices.sort(key=lambda v: v["created"], reverse=True)
    return jsonify(voices=voices, languages=LANGUAGES)


@app.post("/api/voices")
def create_voice():
    owner = client_id()
    name = (request.form.get("name") or "").strip()[:60]
    if not name:
        abort(400, "name is required")
    if request.form.get("consent") != "yes":
        abort(400, "consent is required")
    files = request.files.getlist("samples")
    if not files:
        abort(400, "at least one sample is required")
    vid = uuid.uuid4().hex[:12]
    d = VOICES_DIR / vid
    d.mkdir()
    write_meta(d, {"id": vid, "name": name, "created": time.time(),
                   "consent": True, "owner": owner})
    if save_samples(d, files) == 0:
        shutil.rmtree(d)
        abort(400, "no usable samples (each must be a valid clip of at least 1 second)")
    return jsonify(voice_summary(d))


@app.post("/api/voices/<voice_id>/samples")
def add_samples(voice_id):
    d = voice_dir(voice_id)
    save_samples(d, request.files.getlist("samples"))
    return jsonify(voice_summary(d))


@app.delete("/api/voices/<voice_id>")
def delete_voice(voice_id):
    shutil.rmtree(voice_dir(voice_id))
    return jsonify(ok=True)


@app.post("/api/synthesize")
def synthesize():
    body = request.get_json(force=True)
    d = voice_dir(str(body.get("voice_id", "")))
    text = (body.get("text") or "").strip()
    lang = body.get("language", "en")
    if not text:
        abort(400, "text is required")
    if len(text) > MAX_TEXT:
        abort(400, f"text is limited to {MAX_TEXT} characters")
    if lang not in LANGUAGES:
        abort(400, "unsupported language")

    model = get_model()
    gpt, spk = get_latents(d)
    device = next(model.parameters()).device
    with _model_lock:
        out = model.inference(
            text, lang, gpt.to(device), spk.to(device),
            temperature=float(body.get("temperature", 0.7)),
            enable_text_splitting=True,
        )
    name = f"{d.name}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(OUTPUT_DIR / name, out["wav"], 24000)
    return jsonify(url=f"/outputs/{name}")


@app.get("/outputs/<name>")
def output(name):
    path = (OUTPUT_DIR / name).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.exists():
        abort(404)
    return send_file(path, mimetype="audio/wav")


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def err(e):
    return jsonify(error=getattr(e, "description", str(e))), e.code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    app.run(host=host, port=port, threaded=True)
