from transformers import pipeline

# Load a zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_activities_zsc(text, candidate_activities):
    result = classifier(text, candidate_activities, multi_label=True)
    return [activity for activity, score in zip(result["labels"], result["scores"]) if score > 0.5]

text = "I went to Paris, where I went to the Eiffel Tower, I visited the Louvre, which is a famous museum, and I went swimming near the beach."
candidate_activities = ["hiking", "swimming", "sightseeing", "skiing", "snorkeling", "road trip", "museum"]

print(extract_activities_zsc(text, candidate_activities))