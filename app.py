from src.vad_recorder import SileroVADRecorder
from src.whisper_cpp import transcribe_with_whisper_cpp


def main():
    # device=None -> auto-pick a microphone via a reliable host API and use its
    # native sample rate. On this PC the Bluetooth earbud mic is only reachable
    # through the Windows "Default recording device", so make sure your earbuds
    # are set as the Default recording device in Windows Sound settings.
    # Pass an index (e.g. device=12) to force a specific mic instead.
    recorder = SileroVADRecorder()

    print("Voice agent started. Press CTRL + C to quit.")
    print("Tip: set your earbuds as the Windows DEFAULT recording device.")
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