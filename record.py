import sounddevice as sd
import numpy as np
import wave
import keyboard


# Record Audio
def record_audio_until_interrupt(filename="input.wav", samplerate=16000):
    print("Recording... Press 'q' to stop.")

    # Prepare an empty list to store audio chunks
    audio_chunks = []
    block_size = 1024  # Size of each audio block to record

    # Callback function to handle audio chunks
    def callback(indata, frames, time, status):
        if status:
            print(f"Audio recording error: {status}")
        audio_chunks.append(indata.copy())

    try:
        # Open audio stream
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            callback=callback,
            blocksize=block_size,
        ):
            while not keyboard.is_pressed("q"):  # Keep recording until 'q' is pressed
                sd.sleep(100)  # Sleep to avoid busy waiting

            print("\nRecording stopped. Saving audio...")

        # Concat all recorded audio chunks into a single numpy array
        audio_data = np.concatenate(audio_chunks, axis=0)

        # Save the audio data to a WAV file
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())

        print(f"Audio saved as {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Execute the function for audio recording
record_audio_until_interrupt()
