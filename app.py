from src.vad_recorder import SileroVADRecorder
from src.whisper_cpp import transcribe_with_whisper_cpp


def main():
    # Choosing the mic:
    #   device=None          -> auto-pick; prefers a mic that actually has signal
    #   device="onenus"      -> pin by NAME (survives Bluetooth index shuffles)
    #   device=12            -> pin by exact index
    # First run `python diagnose_mic.py`, speak when prompted, and use the name
    # of the device that prints "*** HEARS YOUR VOICE ***" below.
    #
    # clean_audio=False: keep the RAW mic audio. On a quiet/low-level mic the
    # denoiser can strip the faint speech, so we leave it off for reliable
    # transcription. Set True again only if you have loud, clear input with
    # steady background noise you want removed.
    recorder = SileroVADRecorder(device=None, clean_audio=False)

    print("Voice agent started. Press CTRL + C to quit.")
    print("Watch the 'Level: .. dBFS' line: aim for -20 to -6 dBFS. If it's")
    print("around -35 dBFS or lower, raise the Windows mic Level + Microphone Boost.")
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