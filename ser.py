from transformers import pipeline

# Load the fine-tuned model and feature extractor
pipe = pipeline("audio-classification", model="Khoa/w2v-speech-emotion-recognition")

# Path to your audio file
audio_file = "C:\\Users\\jevan\\Desktop\\TUDelft\\DSAIT4000ConversationalAgents\\DSAIT4065_TravelAgent\\TestBestand.wav"

# Perform emotion classification
predictions = pipe(audio_file)

# Map predictions to real emotion labels
label_map = {
    "LABEL_0": "sadness",
    "LABEL_1": "angry",
    "LABEL_2": "disgust",
    "LABEL_3": "fear",
    "LABEL_4": "happy",
    "LABEL_5": "neutral"
}

# Convert predictions to readable labels
mapped_predictions = [
    {"score": pred["score"], "label": label_map[pred["label"]]}
    for pred in predictions
]

# Display results
print(mapped_predictions)
