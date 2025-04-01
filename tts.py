from google.cloud import texttospeech
import os
from playsound import playsound
import tempfile
import pygame

class TextToSpeech:
    def __init__(self, credentials_path="API_key.json"):
        # Set credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # Initialize the client
        self.client = texttospeech.TextToSpeechClient()
        
        # Initialize pygame mixer for audio state tracking
        pygame.mixer.init()
        
        # Set default voice parameters
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Chirp3-HD-Leda",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Set default audio configuration
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        
        # Keep track of current audio file
        self.current_audio = None

    def stop_current_audio(self):
        """Stop any currently playing audio and clean up."""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        if self.current_audio and os.path.exists(self.current_audio):
            try:
                os.remove(self.current_audio)
                self.current_audio = None
            except:
                pass

    def play(self, text):
        # Stop any currently playing audio
        self.stop_current_audio()
        
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Perform the text-to-speech request
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(response.audio_content)
            temp_path = temp_file.name
            self.current_audio = temp_path

        # Play the audio
        try:
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            # Clean up the temporary file
            pygame.mixer.music.unload()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.current_audio = None

# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.play("Movies, oh my gosh, I just just absolutely love them. They're like time machines taking you to different worlds and landscapes, and um, and I just can't get enough of it.")