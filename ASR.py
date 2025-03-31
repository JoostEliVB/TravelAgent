import sounddevice as sd
import numpy as np
import queue
import sys
import time
import speech_recognition as sr
import io
import wave

def record_until_silence(fs=16000, threshold=500, silence_duration=1.0, max_recording=30.0):
    """
    Records audio from the microphone until a period of silence is detected.

    Parameters:
      fs (int): Sample rate (default 16000 Hz).
      threshold (int): Amplitude threshold below which audio is considered silent.
      silence_duration (float): Duration (in seconds) of consecutive silence required to stop recording.
      max_recording (float): Maximum recording duration in seconds.

    Returns:
      numpy.ndarray: Recorded audio data
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
                amplitude = np.abs(data).mean()
                if amplitude < threshold:
                    silence_counter += len(data) / fs
                else:
                    silence_counter = 0.0

            if silence_counter >= silence_duration or (time.time() - start_time) >= max_recording:
                break

    return np.concatenate(audio_data, axis=0)

def transcribe_audio(audio_data, fs=16000):
    """
    Transcribe audio data directly without saving to file.
    
    Parameters:
        audio_data (numpy.ndarray): Audio data to transcribe
        fs (int): Sample rate of the audio
    
    Returns:
        str: Transcribed text
    """
    # Convert audio data to WAV format in memory
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(fs)
        wav_file.writeframes(audio_data.tobytes())
    
    # Create recognizer and transcribe
    r = sr.Recognizer()
    byte_io.seek(0)
    with sr.AudioFile(byte_io) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            print("Transcription:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

if __name__ == "__main__":
    # Record until silence is detected
    audio = record_until_silence()
    # Transcribe the recorded audio directly
    transcription = transcribe_audio(audio)
