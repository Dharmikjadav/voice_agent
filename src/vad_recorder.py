import msvcrt
import threading
from math import gcd

import torch
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, VADIterator

from src.denoise import denoise



DEFAULT_MIC_HINTS = ("hands-free", "headset", "headphone", "earbud", "airdopes", "buds")

PREFERRED_HOSTAPIS = ("Windows WASAPI", "MME", "Windows DirectSound")

DEFAULT_ROUTING_NAMES = ("sound mapper", "primary sound capture")

EXCLUDE_NAMES = ("stereo mix", "what u hear", "wave out mix", "loopback", "aux")


def _probe_open(index, samplerate, channels):
    """Return True if a capture stream on this device can actually start."""
    try:
        with sd.InputStream(device=index, samplerate=samplerate,
                            channels=min(channels, 2), dtype="float32",
                            blocksize=1024) as stream:
            stream.read(256)  # pull a little audio to be sure it's live
        return True
    except Exception:
        return False


def list_input_candidates(hints=DEFAULT_MIC_HINTS):
    """All input devices, ranked best-first for speech capture.

    Ranking tiers (each falls back to the next):
      0. headset/earbud matched by name on a reliable host API (ideal)
      1. the Windows default-recording-device router on a reliable host API
         (Sound Mapper / Primary Capture) -> follows Windows Sound settings
      2. any other device on a reliable host API
      3. anything else, including WDM-KS, as a last resort
    Within a tier: better host API first, then higher native sample rate.
    """
    hostapis = sd.query_hostapis()

    def rank_hostapi(name):
        return PREFERRED_HOSTAPIS.index(name) if name in PREFERRED_HOSTAPIS else len(PREFERRED_HOSTAPIS)

    candidates = []
    for index, d in enumerate(sd.query_devices()):
        if int(d["max_input_channels"]) <= 0:
            continue
        name = d["name"].splitlines()[0]
        if any(x in name.lower() for x in EXCLUDE_NAMES):
            continue  # loopback / system-output capture, not a mic
        ha = hostapis[d["hostapi"]]["name"]
        candidates.append({
            "index": index,
            "name": name,
            "hostapi": ha,
            "rate": int(d["default_samplerate"]),
            "channels": int(d["max_input_channels"]),
        })

    def is_hint(name):
        return any(h in name.lower() for h in hints)

    def is_default_router(name):
        return any(k in name.lower() for k in DEFAULT_ROUTING_NAMES)

    def tier(c):
        working = c["hostapi"] in PREFERRED_HOSTAPIS
        if is_hint(c["name"]) and working:
            return 0
        if is_default_router(c["name"]) and working:
            return 1
        return 2 if working else 3

    candidates.sort(key=lambda c: (
        tier(c),
        0 if is_hint(c["name"]) else 1,   # prefer the real headset over Line In
        rank_hostapi(c["hostapi"]),
        -c["rate"],
    ))
    return candidates


def find_input_device(hints=DEFAULT_MIC_HINTS, probe=True):
    """Pick the best input device that can actually start a stream.

    Returns a candidate dict {index, name, hostapi, rate, channels}.
    """
    candidates = list_input_candidates(hints)
    if not candidates:
        raise RuntimeError("No input (microphone) device found. Connect a mic and retry.")
    if probe:
        for c in candidates:
            if _probe_open(c["index"], c["rate"], c["channels"]):
                return c
    return candidates[0]


class SileroVADRecorder:
    def __init__(self, vad_sample_rate=16000, chunk_size=2048, device=None,
                 mic_hints=DEFAULT_MIC_HINTS, clean_audio=True):
        self.vad_sample_rate = vad_sample_rate
        self.chunk_size = chunk_size
        self.mic_hints = mic_hints
        # Run the captured turn through the noise-reduction chain (high-pass +
        # spectral gate + noise gate) before saving. Turn off with
        # clean_audio=False to keep the raw mic audio.
        self.clean_audio = clean_audio

        # Build the ordered list of devices to try. An explicit `device` index
        # pins to that one; otherwise auto-rank and probe.
        if device is None:
            self.candidates = list_input_candidates(mic_hints)
            if not self.candidates:
                raise RuntimeError("No input (microphone) device found. Connect a mic and retry.")
        else:
            d = sd.query_devices(device)
            if int(d["max_input_channels"]) <= 0:
                raise ValueError(
                    f"Device {device} ('{d['name'].splitlines()[0]}') has no input "
                    f"channels - it is an output device, not a microphone."
                )
            hostapis = sd.query_hostapis()
            self.candidates = [{
                "index": device,
                "name": d["name"].splitlines()[0],
                "hostapi": hostapis[d["hostapi"]]["name"],
                "rate": int(d["default_samplerate"]),
                "channels": int(d["max_input_channels"]),
            }]

        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.model,
            threshold=0.30,
            sampling_rate=16000,
            min_silence_duration_ms=1000,
            speech_pad_ms=500,
        )

        self.is_recording = False
        self.audio_buffer = []
        self.vad_buffer = np.array([], dtype=np.float32)
        self.pre_speech_buffer = []
        self._stop = threading.Event()

        # Per-device fields, set by _apply_device().
        self.device = None
        self.device_name = None
        self.hostapi = None
        self.mic_sample_rate = None
        self.channels = None
        self._up = self._down = 1

    def _apply_device(self, c):
        self.device = c["index"]
        self.device_name = c["name"]
        self.hostapi = c["hostapi"]
        self.mic_sample_rate = c["rate"]
        self.channels = min(c["channels"], 2)
        g = gcd(self.mic_sample_rate, self.vad_sample_rate)
        self._up = self.vad_sample_rate // g
        self._down = self.mic_sample_rate // g

    def _to_16k(self, indata):
        mono = np.mean(indata, axis=1).copy()
        if self._up == self._down:
            return mono.astype(np.float32)
        return resample_poly(mono, self._up, self._down).astype(np.float32)

    def _make_callback(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)

            mono_16000 = self._to_16k(indata)
            self.vad_buffer = np.concatenate([self.vad_buffer, mono_16000])

            while len(self.vad_buffer) >= 512:
                vad_chunk = self.vad_buffer[:512]
                self.vad_buffer = self.vad_buffer[512:]

                # keep small audio before speech starts
                if not self.is_recording:
                    self.pre_speech_buffer.append(vad_chunk)
                    if len(self.pre_speech_buffer) > 10:
                        self.pre_speech_buffer.pop(0)

                audio_tensor = torch.from_numpy(vad_chunk)
                speech = self.vad_iterator(audio_tensor, return_seconds=True)

                if speech:
                    if "start" in speech:
                        print("Speech started")
                        self.is_recording = True
                        # add previous chunks so first word is not cut
                        self.audio_buffer = self.pre_speech_buffer.copy()

                    if "end" in speech:
                        print("Speech ended")
                        self.is_recording = False
                        self._stop.set()
                        return

                if self.is_recording:
                    self.audio_buffer.append(vad_chunk)

        return callback

    def _open_stream(self):
        """Try candidates in order; return the first stream that starts.

        Skips devices that fail to start (e.g. broken WDM-KS / disconnected
        Bluetooth) and reports a clear, actionable error if none work.
        """
        errors = []
        for c in self.candidates:
            self._apply_device(c)
            try:
                stream = sd.InputStream(
                    device=self.device,
                    samplerate=self.mic_sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    callback=self._make_callback(),
                )
                stream.start()
                return stream
            except Exception as e:
                first = str(e).splitlines()[0]
                errors.append(f"  [{c['hostapi']}] {c['name']}: {first}")
                continue

        msg = ["Could not start any microphone stream. Tried:"]
        msg += errors
        msg.append("")
        msg.append("Fix: open Windows Sound settings -> Input, set your earbuds as the")
        msg.append("Default recording device (and enable 'Hands-Free Telephony' for them),")
        msg.append("then run again. The app will capture via the default-input router.")
        raise RuntimeError("\n".join(msg))

    def record_until_silence(self, output_path="audio/user_input.wav"):
        self.vad_iterator.reset_states()
        self.is_recording = False
        self.audio_buffer = []
        self.vad_buffer = np.array([], dtype=np.float32)
        self.pre_speech_buffer = []
        self._stop.clear()

        stream = self._open_stream()
        router = any(k in self.device_name.lower() for k in DEFAULT_ROUTING_NAMES)
        print(f"Using device: {self.device_name} [{self.hostapi}] @ {self.mic_sample_rate} Hz")
        if router:
            print("(routing to your Windows Default recording device)")
        print("Listening... speak now, or press any key to stop.")

        try:
            # Poll until VAD signals end-of-speech OR the user presses a key.
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    msvcrt.getch()  # consume the keystroke
                    print("Stopped by key press.")
                    self.is_recording = False
                    self._stop.set()
                    break
                sd.sleep(50)
        finally:
            stream.stop()
            stream.close()

        # Stream is closed here, so the callback no longer touches the buffers.
        chunks = self.audio_buffer if self.audio_buffer else self.pre_speech_buffer
        if not chunks:
            print("No speech captured. (If you spoke, your mic level may be silent - "
                  "check that your earbuds are the Windows Default recording device.)")
            return None

        audio_data = np.concatenate(chunks)

        # Guard against a "working" device that carries no real signal (e.g. a
        # Line In jack with nothing plugged in, or a mic Windows isn't routing).
        # Digital silence sits near -inf dBFS; a real mic always has some floor.
        peak = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
        if peak < 8e-4:
            print(
                f"WARNING: '{self.device_name}' [{self.hostapi}] produced no "
                f"signal (peak {20 * np.log10(peak + 1e-9):.0f} dBFS)."
            )
            print("  Your microphone is not reaching this input device.")
            print("  Run:  python diagnose_mic.py   to find a mic that hears you,")
            print("  or set your earbuds/headset as the Windows DEFAULT recording device.")
            return None

        if self.clean_audio:
            audio_data = denoise(audio_data, self.vad_sample_rate)

        write(
            output_path,
            self.vad_sample_rate,
            (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16),
        )

        print(f"Saved audio: {output_path}")
        return output_path
