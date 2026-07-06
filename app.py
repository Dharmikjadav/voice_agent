from src.vad_recorder import SileroVADRecorder
from src.whisper_cpp import transcribe_with_whisper_cpp, MIN_CONFIDENCE


def main():
    # Choosing the mic:
    #   device=None          -> auto-pick; prefers a mic that actually has signal
    #   device="onenus"      -> pin by NAME (survives Bluetooth index shuffles)
    #   device=12            -> pin by exact index
    # First run `python diagnose_mic.py`, speak when prompted, and use the name
    # of the device that prints "*** HEARS YOUR VOICE ***" below.
    #
    # clean_audio=True: apply noise reduction, but only to loud-enough takes.
    # On a quiet mic the denoiser can strip faint speech, so quiet takes are
    # kept raw automatically (see DENOISE_MIN_PEAK in vad_recorder.py).
    recorder = SileroVADRecorder(device=None, clean_audio=True)

    print("Voice agent started. Press CTRL + C to quit.")
    print("Watch the 'Level: .. dBFS' line: aim for -20 to -6 dBFS. If it's")
    print("around -35 dBFS or lower, raise the Windows mic Level + Microphone Boost.")
    print("Speak, then go silent (~1s) or press any key to end each turn.\n")

    try:
        while True:
            audio_path = recorder.record_until_silence(
                output_path="audio/user_input.wav"
            )

            if audio_path is None:
                continue

            text, conf = transcribe_with_whisper_cpp(audio_path)

            if not text:
                print("No text detected by Whisper.")
                print("\nListening again...\n")
                continue

            # Quality gate: reject unreliable transcriptions and ask to repeat.
            if conf is not None and conf < MIN_CONFIDENCE:
                print(f"Low confidence ({conf:.0%}) - please repeat.")
                print("\nListening again...\n")
                continue

            conf_txt = f"  [confidence {conf:.0%}]" if conf is not None else ""
            print("\nUser said:")
            print(text + conf_txt)
            print("\nListening again...\n")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.close()  # release the microphone stream cleanly


if __name__ == "__main__":
    main()