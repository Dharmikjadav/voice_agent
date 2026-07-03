"""Lightweight, dependency-free speech denoiser.

Cleans a 16 kHz mono float32 recording using only numpy/scipy (already
required by the project) so there is nothing new to install. Three stages,
in order:

  1. High-pass filter  - removes sub-80 Hz rumble / mains hum.
  2. Spectral gating    - estimates the stationary noise spectrum from the
                          quietest frames of the clip and attenuates any
                          time-frequency bin that sits near that noise
                          floor (the same idea as the `noisereduce` lib).
  3. Noise gate         - fully ducks whole frames that are still below the
                          speech threshold, so pauses become silent instead
                          of "hiss".

Tuned for the ~30 dB SNR / -50 dBFS noise floor measured on this project's
recordings. All knobs are keyword args if you want to dial it back.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft


def highpass(x, sr, cutoff=80.0, order=4):
    """Remove low-frequency rumble/hum below `cutoff` Hz (zero-phase)."""
    if cutoff <= 0:
        return x
    sos = butter(order, cutoff / (sr / 2.0), btype="high", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _smooth2d(mask, t_span=3, f_span=3):
    """Box-smooth the gate mask over time & frequency to kill musical noise."""
    if t_span > 1:
        k = np.ones(t_span, dtype=np.float32) / t_span
        mask = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, mask)
    if f_span > 1:
        k = np.ones(f_span, dtype=np.float32) / f_span
        mask = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, mask)
    return mask


def spectral_gate(x, sr, n_fft=512, hop=128, noise_pct=0.15,
                  n_std=1.5, reduction_db=18.0):
    """Attenuate stationary background noise via spectral gating.

    Noise profile is learned from the quietest `noise_pct` of frames, so it
    needs no separate "noise sample" - the pauses in the clip are enough.
    `reduction_db` caps how hard noise-only bins are ducked (not to silence,
    which would sound unnatural).
    """
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                   boundary="zeros", padded=True)
    mag = np.abs(Z)
    phase = np.angle(Z)

    frame_energy = mag.mean(axis=0)
    k = max(1, int(noise_pct * mag.shape[1]))
    quiet_idx = np.argsort(frame_energy)[:k]
    noise_mean = mag[:, quiet_idx].mean(axis=1, keepdims=True)
    noise_std = mag[:, quiet_idx].std(axis=1, keepdims=True)
    thresh = noise_mean + n_std * noise_std

    floor = 10.0 ** (-reduction_db / 20.0)          # e.g. -18 dB -> ~0.126
    mask = np.where(mag >= thresh, 1.0, floor).astype(np.float32)
    mask = _smooth2d(mask, t_span=3, f_span=3)
    mask = np.clip(mask, floor, 1.0)

    Zc = mag * mask * np.exp(1j * phase)
    _, xr = istft(Zc, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                  boundary=True)
    return xr[: len(x)].astype(np.float32)


def noise_gate(x, sr, frame_ms=20, margin_db=8.0, attn_db=25.0,
               attack_ms=10, release_ms=120):
    """Duck whole frames that sit near the noise floor (silences the pauses).

    Uses a smoothed per-frame RMS envelope with attack/release so it doesn't
    chop word onsets. `margin_db` above the estimated noise floor is the
    open threshold; quiet frames are attenuated by `attn_db` (not zeroed).
    """
    frame = max(1, int(frame_ms * sr / 1000))
    n = len(x)
    n_frames = int(np.ceil(n / frame))
    padded = np.pad(x, (0, n_frames * frame - n))
    frames = padded.reshape(n_frames, frame)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    noise_floor = np.mean(np.sort(db)[: max(1, int(0.2 * n_frames))])
    open_thresh = noise_floor + margin_db

    target = np.where(db >= open_thresh, 1.0, 10.0 ** (-attn_db / 20.0))

    # attack/release smoothing of the gain envelope
    a_atk = np.exp(-1.0 / max(1, attack_ms / frame_ms))
    a_rel = np.exp(-1.0 / max(1, release_ms / frame_ms))
    gain = np.empty_like(target)
    g = target[0]
    for i, tgt in enumerate(target):
        a = a_atk if tgt > g else a_rel
        g = a * g + (1 - a) * tgt
        gain[i] = g

    gained = (frames * gain[:, None]).reshape(-1)[:n]
    return gained.astype(np.float32)


def denoise(x, sr=16000, hp_cutoff=80.0, spectral=True, gate=True):
    """Full clean-up chain for a mono float32 speech clip in [-1, 1]."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x = highpass(x, sr, cutoff=hp_cutoff)
    if spectral:
        x = spectral_gate(x, sr)
    if gate:
        x = noise_gate(x, sr)
    # guard against any clipping introduced by filtering
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x
