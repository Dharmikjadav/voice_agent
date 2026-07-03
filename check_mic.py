import sounddevice as sd

print("Default devices [input, output]:", sd.default.device)
print()

devices = sd.query_devices()

for i, d in enumerate(devices):
    if d["max_input_channels"] > 0:
        print(
            f"{i}: {d['name'].splitlines()[0]} "
            f"| in_ch={d['max_input_channels']} "
            f"| rate={int(d['default_samplerate'])} Hz"
        )
