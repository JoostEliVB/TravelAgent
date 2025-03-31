from google.cloud import texttospeech
import os
from playsound import playsound
import tempfile

class TextToSpeech:
    def __init__(self, credentials_path="API_key.json"):
        # Set credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # Initialize the client
        self.client = texttospeech.TextToSpeechClient()
        
        # Set default voice parameters
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Chirp3-HD-Leda",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Set default audio configuration
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    def play(self, text):
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Perform the text-to-speech request
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.audio_content)
            temp_path = temp_file.name

        # Play the audio
        playsound(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)

# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.play("Movies, oh my gosh, I just just absolutely love them. They're like time machines taking you to different worlds and landscapes, and um, and I just can't get enough of it.")