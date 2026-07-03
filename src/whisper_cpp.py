import subprocess
import os
import re


def clean_whisper_output(output: str) -> str:
    lines = output.splitlines()
    cleaned = []

    for line in lines:
        # remove timestamps like: [00:00:00.000 --> 00:00:02.000]
        line = re.sub(r"\[.*?-->\s*.*?\]", "", line).strip()

        if line:
            cleaned.append(line)

    return " ".join(cleaned).strip()


def transcribe_with_whisper_cpp(audio_path):
    whisper_exe = r"whisper_cpp\whisper-cli.exe"
    model_path = r"whisper_cpp\models\ggml-base.en.bin"

    if not os.path.exists(whisper_exe):
        raise FileNotFoundError(f"whisper-cli.exe not found: {whisper_exe}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Whisper model not found: {model_path}")

    command = [
        whisper_exe,
        "-m", model_path,
        "-f", audio_path,
        "-nt",
        "-np"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        print("Whisper error:")
        print(result.stderr)
        return None

    return clean_whisper_output(result.stdout)