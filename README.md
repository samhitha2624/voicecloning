---
title: Voice Clone Studio
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Voice Clone Studio

Local web app: upload or record as many samples of a voice as you like, and it builds
one voice profile from all of them (Coqui XTTS-v2), then speaks any text in that voice.

## Setup (Python 3.10–3.12)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

The first generation downloads the XTTS-v2 model (~1.8 GB) and asks you in the terminal to
accept its license (Coqui Public Model License — non-commercial use only).

## Tips for a good clone
- 1–5 minutes of clean speech total; one speaker only, no music or background noise.
- Vary tone: questions, excitement, calm sentences. Use the built-in reading prompts.
- Adding samples later recomputes the voice from all samples.

## Notes
- Runs on CPU if there is no suitable GPU; expect ~10–60 s per sentence on CPU.
- Only clone voices you own or have explicit permission to use.
