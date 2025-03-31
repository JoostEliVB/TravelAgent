import sounddevice as sd
import numpy as np
import queue
import sys
import time
import scipy.io.wavfile as wavfile
import speech_recognition as sr
from transformers import pipeline


def record_until_silence(fs=16000, threshold=500, silence_duration=1.0, max_recording=30.0):
    """
    Records audio from the microphone until a period of silence is detected.

    Parameters:
      fs (int): Sample rate (default 16000 Hz).
      threshold (int): Amplitude threshold below which audio is considered silent.
      silence_duration (float): Duration (in seconds) of consecutive silence required to stop recording.
      max_recording (float): Maximum recording duration in seconds.

    Returns:
      (numpy.ndarray, int): Recorded audio and sample rate.
    """
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("Recording... Speak now. (Recording will stop after {:.1f} sec of silence)".format(silence_duration))
    audio_data = []
    silence_counter = 0.0
    start_time = time.time()

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        while True:
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                data = None

            if data is not None:
                audio_data.append(data)
                # Calculate the mean absolute amplitude
                amplitude = np.abs(data).mean()
                # If the amplitude is below threshold, increment silence duration
                if amplitude < threshold:
                    silence_counter += len(data) / fs
                else:
                    silence_counter = 0.0

            # Stop if silence is long enough or if max recording duration is reached
            if silence_counter >= silence_duration or (time.time() - start_time) >= max_recording:
                break

    audio = np.concatenate(audio_data, axis=0)
    return audio, fs


def save_audio(filename, audio, fs):
    """Save the recorded audio to a WAV file."""
    wavfile.write(filename, fs, audio)
    print("Recording complete. Saved to", filename)


def speech_to_text_from_file(filename="output.wav"):
    """Transcribe speech from a WAV file using Google Speech Recognition."""
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print("Transcription:", text)
        return text
    except Exception as e:
        print("Error during transcription:", e)
        return ""


def recognize_emotion(text):
    """
    Recognize emotion from text using a transformer-based model.
    The model returns scores for several emotions.
    """
    # Initialize the emotion classifier pipeline
    classifier = pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base",
                          return_all_scores=True)
    results = classifier(text)

    # Depending on the model, results may be nested in a list
    if results and isinstance(results, list):
        # If the results are in a nested list, use the first element
        if isinstance(results[0], list):
            results = results[0]
        # Find the label with the highest score
        best = max(results, key=lambda x: x['score'])
        print("Detected emotion:", best['label'], "with confidence", best['score'])
    else:
        print("Could not detect emotion.")


if __name__ == "__main__":
    # # Record until silence is detected
    # audio, fs = record_until_silence()
    # # Save the audio to a file
    # save_audio("output.wav", audio, fs)
    # # Transcribe the recorded audio
    # transcription = speech_to_text_from_file("output.wav")
    # # If transcription is successful, perform emotion recognition on the text
    # if transcription:
    #     recognize_emotion(transcription)
    #
    #
    recognize_emotion("I like to go to the beach, but dislike the mountains")
    # recognize_emotion("I really like to go to the beach")
    # recognize_emotion("I hate to go to the beach")
    # recognize_emotion("I really hate to go to the beach")
    # recognize_emotion("Are you going to the beach?")
    # recognize_emotion("I like to go to the mountains")
    # recognize_emotion("I am in love with the beach")