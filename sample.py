import sounddevice as sd
import numpy as np

DEVICE = 1
SAMPLE_RATE = 44100

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata) * 10
    print("Volume:", round(volume, 2))

with sd.InputStream(
    device=DEVICE,
    samplerate=SAMPLE_RATE,
    channels=2,
    dtype="float32",
    callback=callback
):
    print("Speak into mic...")
    sd.sleep(10000)