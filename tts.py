from google.cloud import texttospeech
import os
from playsound import playsound
import tempfile

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "API_key.json"

def text_to_speech_play(text):
    # Initialize the client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Chirp3-HD-Leda",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Select the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
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
text_to_speech_play("Movies, oh my gosh, I just just absolutely love them. They're like time machines taking you to different worlds and landscapes, and um, and I just can't get enough of it.")