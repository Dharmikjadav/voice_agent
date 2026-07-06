import subprocess
import os
import re
import json


WHISPER_EXE = r"whisper_cpp\whisper-cli.exe"

# ---- Whisper model: swap this one line to change accuracy/speed ----
# ggml-base.en.bin   142 MB  fast          (current)
# ggml-small.en.bin  466 MB  slower, more accurate  <- better for noisy mic audio
# Download models from:
#   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/<name>.bin
# and place them in whisper_cpp\models\.
MODEL_PATH = r"whisper_cpp\models\ggml-base.en.bin"

# Quality gate: mean per-token probability below this is treated as an
# unreliable transcription (see app.py, which asks the user to repeat).
# Clear speech usually averages ~0.8-0.95; uncertain ~0.4-0.6. Tune as needed.
MIN_CONFIDENCE = 0.60


def confidence_from_json(json_path):
    """Mean per-token probability from a whisper `-ojf` JSON, in [0, 1].

    Returns None if the file is missing/unparseable or has no usable tokens, so
    a JSON-format surprise never silently blocks every transcription.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        probs = []
        for seg in data.get("transcription", []):
            for tok in seg.get("tokens", []):
                text = (tok.get("text") or "").strip()
                # skip special/timestamp tokens like [_BEG_], [_TT_123]
                if not text or text.startswith("["):
                    continue
                p = tok.get("p")
                if isinstance(p, (int, float)):
                    probs.append(float(p))
        if not probs:
            return None
        return sum(probs) / len(probs)
    except Exception:
        return None


def clean_whisper_output(output: str) -> str:
    lines = output.splitlines()
    cleaned = []

    for line in lines:
        # Remove timestamps
        line = re.sub(r"\[.*?-->\s*.*?\]", "", line).strip()

        if line:
            cleaned.append(line)

    return " ".join(cleaned).strip()


def transcribe_with_whisper_cpp(audio_path, model_path=MODEL_PATH):
    whisper_exe = WHISPER_EXE

    if not os.path.exists(whisper_exe):
        raise FileNotFoundError(f"whisper-cli.exe not found: {whisper_exe}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Whisper model not found: {model_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    json_base = os.path.splitext(audio_path)[0]      # audio\user_input(.json)
    json_path = json_base + ".json"

    command = [
        whisper_exe,
        "-m", model_path,
        "-f", audio_path,
        "-l", "en",
        # Light context prompt: nudges spelling/casing without a long prompt
        # that can make Whisper hallucinate words on short or noisy clips.
        "--prompt", "Natural English conversation with an AI voice assistant.",
        "-ojf",                 # full JSON with per-token probabilities
        "-of", json_base,       # -> <audio_base>.json
        "-nt",
        "-np",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        # Show everything so real failures (e.g. a missing DLL, exit 127) are
        # visible instead of a blank line.
        print(f"Whisper failed (exit {result.returncode}):")
        err = (result.stderr or "").strip() or (result.stdout or "").strip()
        print(err or "(no output - likely a missing DLL next to whisper-cli.exe)")
        return None, None

    # Extract the transcription text and its confidence.
    text = clean_whisper_output(result.stdout)
    confidence = confidence_from_json(json_path)

    if not text:
        print("No Whisper text found")
        return None, None

    return text, confidence