from src.vad_recorder import SileroVADRecorder
from src.whisper_cpp import transcribe_with_whisper_cpp


def main():
    # Choosing the mic:
    #   device=None          -> auto-pick; prefers a mic that actually has signal
    #   device="onenus"      -> pin by NAME (survives Bluetooth index shuffles)
    #   device=12            -> pin by exact index
    # First run `python diagnose_mic.py`, speak when prompted, and use the name
    # of the device that prints "*** HEARS YOUR VOICE ***" below.
    recorder = SileroVADRecorder()

    print("Voice agent started. Press CTRL + C to quit.")
    print("Tip: if it records silence, run 'python diagnose_mic.py' to find your mic.")
    print("Speak, then go silent (~1s) or press any key to end each turn.\n")

    while True:
        audio_path = recorder.record_until_silence(
            output_path="audio/user_input.wav"
        )

        if audio_path is None:
            continue

        text = transcribe_with_whisper_cpp(audio_path)

        if not text:
            print("No text detected by Whisper.")
            print("\nListening again...\n")
            continue

        print("\nUser said:")
        print(text)
        print("\nListening again...\n")


if __name__ == "__main__":
    main()