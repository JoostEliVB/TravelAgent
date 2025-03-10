import librosa
import torch
import numpy as np
import queue
import sys
import time
import sounddevice as sd
import scipy.io.wavfile as wavfile
import speech_recognition as sr
from transformers import pipeline, Wav2Vec2FeatureExtractor, HubertForSequenceClassification


def record_until_silence(fs=16000, threshold=500, silence_duration=5.0, max_recording=30.0):
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


def predict_speech_emotion(audio_path):
    """
    Recognize emotion from speech using a pre-trained Hubert model.
    Returns the predicted emotion and confidence scores.
    """
    try:
        # Load and preprocess the audio
        speech, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Ensure the audio is not empty
        if len(speech) == 0:
            raise ValueError("Audio file is empty or could not be loaded.")

        # Trim silence from the beginning and end
        speech, _ = librosa.effects.trim(speech, top_db=20)

        # Normalize audio to avoid extreme amplitude values
        speech = librosa.util.normalize(speech)

        # Load the model and feature extractor
        model_name = "superb/hubert-base-superb-er"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = HubertForSequenceClassification.from_pretrained(model_name)

        # Extract features
        inputs = feature_extractor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,  # Set max_length to 10 seconds
            truncation=True  # Truncate audio longer than max_length
        )

        # Predict emotion
        with torch.no_grad():
            outputs = model(**inputs)

        # Get probabilities and labels
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        emotion = model.config.id2label[predicted_class]
        confidence = probs[0][predicted_class].item()

        return emotion, confidence

    except Exception as e:
        print(f"Error in speech emotion recognition: {e}")
        return "Unknown", 0.0

def recognize_text_emotion(text):
    """Recognize emotion from text using transformer model"""
    try:
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        results = classifier(text)

        if results and isinstance(results, list):
            return max(results[0], key=lambda x: x['score'])
        return None
    except Exception as e:
        print(f"Error in text emotion recognition: {e}")
        return None


if __name__ == "__main__":
    # Record and save audio
    audio, fs = record_until_silence()
    save_audio("output.wav", audio, fs)

    # Speech-based emotion recognition
    speech_emotion, confidence = predict_speech_emotion("output.wav")
    print(f"\nAudio-based emotion: {speech_emotion}(Confidence: {confidence:.2f})")

    # Text-based emotion recognition
    transcription = speech_to_text_from_file("output.wav")
    if transcription:
        text_emotion = recognize_text_emotion(transcription)
        if text_emotion:
            print(f"Text-based emotion: {text_emotion['label']} (confidence: {text_emotion['score']:.2f})")

    print("\nAnalysis complete. Note: These results might differ because:")
    print("- Audio analysis detects vocal characteristics (tone, pitch, rhythm)")
    print("- Text analysis focuses on semantic content and word choice)")
    print("- Combined analysis provides more comprehensive insights")