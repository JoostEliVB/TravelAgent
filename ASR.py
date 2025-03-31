import sounddevice as sd
import numpy as np
import queue
import sys
import time
import speech_recognition as sr

class SpeechToText:
    def __init__(self, fs=16000, threshold=500, silence_duration=10, max_recording=30.0):
        """
        Initialize the SpeechToText class.
        
        Parameters:
            fs (int): Sample rate (default 16000 Hz)
            threshold (int): Amplitude threshold for silence detection
            silence_duration (float): Duration of silence to stop recording
            max_recording (float): Maximum recording duration in seconds
        """
        self.fs = fs
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.max_recording = max_recording
        self.recognizer = sr.Recognizer()

    def record_until_silence(self):
        """Records audio from the microphone until silence is detected."""
        q = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        print("Recording... Speak now. (Recording will stop after {:.1f} sec of silence)".format(self.silence_duration))
        audio_data = []
        silence_counter = 0.0
        start_time = time.time()

        with sd.InputStream(samplerate=self.fs, channels=1, dtype='int16', callback=callback):
            while True:
                try:
                    data = q.get(timeout=0.1)
                except queue.Empty:
                    data = None

                if data is not None:
                    audio_data.append(data)
                    amplitude = np.abs(data).mean()
                    if amplitude < self.threshold:
                        silence_counter += len(data) / self.fs
                    else:
                        silence_counter = 0.0

                if silence_counter >= self.silence_duration or (time.time() - start_time) >= self.max_recording:
                    break

        return np.concatenate(audio_data, axis=0)

    def listen_and_transcribe(self):
        """
        Record audio and transcribe it to text directly.
        
        Returns:
            str: Transcribed text
        """
        try:
            # Use the default microphone as source
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
                
            # Try to recognize the speech
            text = self.recognizer.recognize_google(audio)
            print("Transcription:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

# Example usage:
if __name__ == "__main__":
    stt = SpeechToText()
    transcription = stt.listen_and_transcribe()
    print(f"Final transcription: {transcription}")