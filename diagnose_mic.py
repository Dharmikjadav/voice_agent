"""Live microphone diagnostic.

Run in your OWN terminal so you can see the meters and speak on cue:

    .venv\\Scripts\\python.exe diagnose_mic.py

For each unique microphone it (1) measures the quiet-room baseline, then
(2) asks you to speak and checks whether the level rises clearly above that
baseline. Judging the *rise* (not an absolute threshold) means it still
detects a working-but-quiet mic. Ctrl+C skips the current device.
"""

import sys
import time

import numpy as np
import sounddevice as sd

BASELINE_SEC = 1.5
SPEAK_SEC = 3.0
RISE_DB = 8.0            # voice must exceed baseline by this much to count
EXCLUDE = ("stereo mix", "what u hear", "wave out mix", "loopback")
HOSTAPI_RANK = ("Windows WASAPI", "MME", "Windows DirectSound", "Windows WDM-KS")


def unique_input_devices():
    """One entry per physical mic (dedupe by name), best host API kept."""
    has = sd.query_hostapis()
    best = {}
    for i, d in enumerate(sd.query_devices()):
        if int(d["max_input_channels"]) <= 0:
            continue
        name = d["name"].splitlines()[0]
        low = name.lower()
        if any(x in low for x in EXCLUDE) or "sound mapper" in low or "primary sound" in low:
            continue
        ha = has[d["hostapi"]]["name"]
        rank = HOSTAPI_RANK.index(ha) if ha in HOSTAPI_RANK else len(HOSTAPI_RANK)
        entry = (i, name, ha, int(d["default_samplerate"]), min(int(d["max_input_channels"]), 2))
        if name not in best or rank < best[name][0]:
            best[name] = (rank, entry)
    return [v[1] for v in sorted(best.values(), key=lambda x: x[0])]


def meter(rms, width=40):
    db = 20 * np.log10(rms + 1e-9)
    filled = int(np.clip((db + 60) / 60, 0, 1) * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {db:6.1f} dBFS"


def _capture(stream_state, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        sys.stdout.write("\r  " + meter(stream_state["rms"]))
        sys.stdout.flush()
        time.sleep(0.05)


def test_device(index, name, hostapi, rate, ch):
    print(f"\n--- Device {index}: {name}  [{hostapi}] {rate} Hz, {ch}ch ---")
    st = {"rms": 0.0, "peak": 0.0, "collect": []}

    def cb(indata, frames, t, status):
        r = float(np.sqrt(np.mean(indata ** 2)))
        st["rms"] = r
        st["collect"].append(r)

    try:
        with sd.InputStream(device=index, samplerate=rate, channels=ch,
                            dtype="float32", blocksize=1024, callback=cb):
            print("  Stay quiet (measuring room)...")
            st["collect"] = []
            _capture(st, BASELINE_SEC)
            base = np.median(st["collect"]) if st["collect"] else 1e-9

            print("\r  SPEAK NOW...                                        ")
            st["collect"] = []
            _capture(st, SPEAK_SEC)
            voice = np.max(st["collect"]) if st["collect"] else 1e-9

        sys.stdout.write("\r" + " " * 60 + "\r")
        base_db, voice_db = 20 * np.log10(base + 1e-9), 20 * np.log10(voice + 1e-9)
        rise = voice_db - base_db
        heard = rise >= RISE_DB and voice_db > -45
        print(f"  baseline={base_db:6.1f} dBFS   voice peak={voice_db:6.1f} dBFS   "
              f"rise={rise:5.1f} dB")
        if heard:
            print("  ->  *** HEARS YOUR VOICE ***")
        elif rise >= RISE_DB:
            print("  ->  voice detected but weak - raise the mic Level/Boost in Windows")
        else:
            print("  ->  no clear voice above the room noise")
        return heard, voice_db, index, name
    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * 60 + "\r")
        print("  (skipped)")
        return False, -99, index, name
    except Exception as e:
        print(f"  FAILED to open -> {str(e).splitlines()[0]}")
        return False, -99, index, name


def main():
    print("Default input/output device index:", sd.default.device)
    devs = unique_input_devices()
    if not devs:
        print("\nNo usable microphone devices found.")
        return

    results = []
    try:
        for (i, name, ha, rate, ch) in devs:
            results.append(test_device(i, name, ha, rate, ch))
    except KeyboardInterrupt:
        print("\nStopped.")

    print("\n" + "=" * 52)
    winners = [r for r in results if r[0]]
    if winners:
        best = max(winners, key=lambda r: r[1])
        print("Mic(s) that heard your voice:")
        for _, vdb, idx, name in winners:
            print(f"   index {idx}: {name}  (voice {vdb:.1f} dBFS)")
        print(f"\nBest -> run:  python app.py   (or pin SileroVADRecorder(device={best[2]}))")
    else:
        print("No device clearly heard your voice.")
        print("Most likely the mic Level is too low: Recording tab -> Microphone")
        print("-> Levels -> set to 100 and add Microphone Boost (+20/+30 dB), then retry.")


if __name__ == "__main__":
    main()
