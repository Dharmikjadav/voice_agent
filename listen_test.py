"""Record a few seconds from your microphone and play it straight back, so
you can HEAR whether your voice is being captured and how clean it sounds.

    .venv\\Scripts\\python.exe listen_test.py

It plays the raw recording first, then the denoised version (the same
clean-up the app applies), so you can compare. Wear your earbuds so the
playback doesn't leak back into the mic.
"""

import numpy as np
import sounddevice as sd

from src.denoise import denoise

DURATION = 5  # seconds to record


def dbfs(x):
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20 * np.log10(p + 1e-9), p


def main():
    in_idx, out_idx = sd.default.device
    info = sd.query_devices(in_idx)
    rate = int(info["default_samplerate"])
    ch = min(int(info["max_input_channels"]), 2)
    print(f"Mic:     {info['name'].splitlines()[0]} @ {rate} Hz")
    print(f"Speaker: {sd.query_devices(out_idx)['name'].splitlines()[0]}")

    print(f"\nRecording {DURATION}s -- SPEAK NOW...")
    rec = sd.rec(int(DURATION * rate), samplerate=rate, channels=ch, dtype="float32")
    sd.wait()

    peak_db, peak = dbfs(rec)
    print(f"Captured peak = {peak_db:.1f} dBFS")
    if peak < 8e-4:
        print("  -> SILENT. Mic captured nothing. Raise the mic level in Windows")
        print("     (Recording tab -> Microphone -> Levels -> 100 + Boost) and retry.")
        return
    if peak_db < -30:
        print("  -> Very quiet. It works, but raise the mic Level/Boost in Windows")
        print("     so your voice peaks nearer -20 to -12 dBFS for good transcription.")

    print("\n[1/2] Playing RAW recording...")
    sd.play(rec, samplerate=rate)
    sd.wait()

    mono = rec.mean(axis=1) if rec.ndim > 1 else rec
    clean = denoise(mono, rate)
    print("[2/2] Playing DENOISED recording...")
    sd.play(clean, samplerate=rate)
    sd.wait()

    print("\nDone. If you heard your voice clearly, the mic works -- run app.py.")


if __name__ == "__main__":
    main()
