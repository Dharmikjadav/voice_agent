import msvcrt
import queue
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


def _peak_level(index, samplerate, channels, seconds=0.25):
    """Peak amplitude over a short read; -1.0 if the device cannot open.

    A live mic always has a small noise floor; a dead / unrouted input (an
    unplugged Line In, or a default-router with no default device) returns
    essentially digital zero. This lets us avoid recording pure silence.
    """
    try:
        frames = max(1024, int(seconds * samplerate))
        with sd.InputStream(device=index, samplerate=samplerate,
                            channels=min(channels, 2), dtype="float32",
                            blocksize=1024) as stream:
            data, _ = stream.read(frames)
        return float(np.max(np.abs(data)))
    except Exception:
        return -1.0


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

        # Build the ordered list of devices to try:
        #   device=None       -> auto-rank all mics, preferring one with signal
        #   device="onenus"   -> pin to input devices whose NAME contains this
        #                        (stable across Bluetooth reconnects that shuffle
        #                        indices; find the exact name via diagnose_mic.py)
        #   device=12         -> pin to that exact index
        if device is None:
            self.candidates = list_input_candidates(mic_hints)
            if not self.candidates:
                raise RuntimeError("No input (microphone) device found. Connect a mic and retry.")
            self._prefer_live_candidates()
        elif isinstance(device, str):
            key = device.lower()
            self.candidates = [c for c in list_input_candidates(mic_hints)
                               if key in c["name"].lower()]
            if not self.candidates:
                raise RuntimeError(
                    f"No input device name contains '{device}'. "
                    f"Run 'python diagnose_mic.py' to see the exact mic names."
                )
            self._prefer_live_candidates()
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
        # Samples (at 16 kHz) to drop at the start of each turn while the VAD
        # state settles. Reset per turn when the recorder is armed.
        self._prime_left = 0

        # Persistent-stream machinery. The mic stream is opened ONCE and kept
        # running for the whole session, so no turn ever pays device-open
        # latency (the old cause of the first utterance being missed). The audio
        # thread only detects speech while _armed is set, and hands finished
        # utterances back to the main thread through a queue.
        self._stream = None
        self._armed = threading.Event()
        self._reset_pending = False
        self._force_end = False
        self._utterances = queue.Queue()

        # Per-device fields, set by _apply_device().
        self.device = None
        self.device_name = None
        self.hostapi = None
        self.mic_sample_rate = None
        self.channels = None
        self._up = self._down = 1

        # Prime the model, then open the persistent stream so audio is already
        # flowing before the first turn (kills the cold-start miss).
        self._warmup()
        self._ensure_stream()

    def _warmup(self):
        # Silero's very first inference is the slow, lazy one - run a few dummy
        # 512-sample frames now so it isn't paid mid-utterance.
        dummy = torch.zeros(512, dtype=torch.float32)
        for _ in range(5):
            self.vad_iterator(dummy, return_seconds=True)
        self.vad_iterator.reset_states()

    def _prefer_live_candidates(self, floor=5e-4):
        """Reorder candidates so a mic that actually carries signal wins.

        Groups: 0 = live (noise floor above `floor`), 1 = opens but silent,
        2 = cannot open. Stable within each group, so the existing tier order
        (headset first, etc.) is preserved. This stops the app from picking a
        device that opens fine but records pure silence.
        """
        scored = []
        for pos, c in enumerate(self.candidates):
            level = _peak_level(c["index"], c["rate"], c["channels"])
            group = 2 if level < 0 else (0 if level >= floor else 1)
            scored.append((group, pos, c))
        scored.sort(key=lambda s: (s[0], s[1]))
        self.candidates = [c for _, _, c in scored]

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

    def _finish_utterance(self):
        """Package the captured audio and hand it to the main thread.

        Runs on the audio thread. Clears recording state, resets the VAD for
        the next turn, and disarms so the stream idles until re-armed.
        """
        if self.is_recording and self.audio_buffer:
            utt = np.concatenate(self.audio_buffer)
        else:
            utt = np.zeros(0, dtype=np.float32)  # nothing captured (e.g. early key press)
        self.is_recording = False
        self.audio_buffer = []
        self.vad_iterator.reset_states()
        self._force_end = False
        self._armed.clear()
        self._utterances.put(utt)

    def _make_callback(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)

            # The stream runs continuously; only detect speech while armed.
            if not self._armed.is_set():
                return

            # First armed callback of a turn: clear buffers and VAD state.
            if self._reset_pending:
                self.vad_iterator.reset_states()
                self.is_recording = False
                self.audio_buffer = []
                self.vad_buffer = np.array([], dtype=np.float32)
                self.pre_speech_buffer = []
                self._prime_left = int(0.15 * self.vad_sample_rate)
                self._reset_pending = False

            mono_16000 = self._to_16k(indata)

            # Drop a little audio right after arming so a settling transient
            # doesn't false-trigger the VAD; reaction time covers this gap.
            if self._prime_left > 0:
                drop = min(self._prime_left, len(mono_16000))
                self._prime_left -= drop
                mono_16000 = mono_16000[drop:]
                if mono_16000.size == 0:
                    return

            self.vad_buffer = np.concatenate([self.vad_buffer, mono_16000])

            while len(self.vad_buffer) >= 512:
                vad_chunk = self.vad_buffer[:512]
                self.vad_buffer = self.vad_buffer[512:]

                # Manual stop (key press) finalizes the current audio immediately.
                if self._force_end:
                    self._finish_utterance()
                    return

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
                        self._finish_utterance()
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

    def _ensure_stream(self):
        """Open the persistent capture stream once and keep it running."""
        if self._stream is not None:
            return
        self._stream = self._open_stream()
        router = any(k in self.device_name.lower() for k in DEFAULT_ROUTING_NAMES)
        print(f"Using device: {self.device_name} [{self.hostapi}] @ {self.mic_sample_rate} Hz")
        if router:
            print("(routing to your Windows Default recording device)")

    def _drain_utterances(self):
        try:
            while True:
                self._utterances.get_nowait()
        except queue.Empty:
            pass

    def close(self):
        """Stop and release the microphone stream."""
        self._armed.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def record_until_silence(self, output_path="audio/user_input.wav"):
        self._ensure_stream()

        # Fresh turn: drop anything captured between turns, flush stray keys,
        # then arm. The audio thread does the buffer/VAD reset on its first
        # armed callback (avoids touching VAD state from two threads).
        self._drain_utterances()
        while msvcrt.kbhit():
            msvcrt.getch()
        self._force_end = False
        self._reset_pending = True
        self._armed.set()

        print("Listening... speak now, or press any key to stop.")

        # Wait for the audio thread to deliver a finished utterance, while
        # watching for a key press that forces an early stop.
        utt = None
        while True:
            if msvcrt.kbhit():
                msvcrt.getch()
                print("Stopped by key press.")
                self._force_end = True
            try:
                utt = self._utterances.get(timeout=0.05)
                break
            except queue.Empty:
                continue

        self._armed.clear()  # callback already disarmed; belt and suspenders

        if utt is None or utt.size == 0:
            print("No speech captured. (If you spoke, your mic level may be silent - "
                  "check that your earbuds are the Windows Default recording device.)")
            return None

        audio_data = utt

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

        # Normalize to a consistent level (~-3 dBFS) so Whisper gets a well-
        # scaled signal even when the Windows mic gain is low. Gain is capped so
        # a near-silent take isn't blown up into pure noise.
        cur_peak = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
        if cur_peak > 0:
            gain = min(10 ** (-3.0 / 20.0) / cur_peak, 10 ** (45.0 / 20.0))
            audio_data = audio_data * gain
            print(f"Level: {20 * np.log10(cur_peak + 1e-9):.1f} dBFS -> "
                  f"normalized (+{20 * np.log10(gain):.1f} dB)")

        write(
            output_path,
            self.vad_sample_rate,
            (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16),
        )

        print(f"Saved audio: {output_path}")
        return output_path
