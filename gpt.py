import os
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Local LLM from Hugging Face
model_name = "openai-community/gpt2"  # Update with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text generation pipeline
text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Integrate with LangChain
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an intelligent travel planning assistant. "
    "Your goal is to understand the user's travel preferences and past experiences "
    "to provide personalized vacation recommendations. "
    "Consider their preferred destinations, activities, accommodations, budget, food choices, "
    "and travel style. Offer detailed itinerary suggestions, must-see attractions, hidden gems, "
    "and travel tips based on their interests. "
    "Only store and recall specific activities the user has enjoyed in past trips to improve suggestions. "
    "Do not store general conversations, future plans, or other non-relevant data. "
    "If they haven't provided enough details, ask relevant questions to refine your suggestions. "
    "Once enough details are gathered, generate a customized travel plan based on stored preferences and past trips. "
    "After presenting a trip recommendation, ask if the user is satisfied with it. "
    "If they are not satisfied, ask what part of the trip they would like to change and generate an alternative recommendation based on their feedback."
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Set up Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Set up Local Milvus Connection using SQLite Backend
milvus = Milvus(
    embedding_model,
    collection_name="past_vacation_experiences",  # Renamed to clarify purpose
    connection_args={"uri": "./milvus_local.db"},  # Local file-based database
)


# Create Memory using Milvus as a Retriever, storing only past vacation activities
memory = VectorStoreRetrieverMemory(retriever=milvus.as_retriever(), input_key="past_vacation_experiences")

# Initialize Conversational Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=MessagesPlaceholder([system_prompt])
)

# Function to generate trip recommendation
def generate_trip_recommendation():
    recommendation_prompt = "Based on the user's travel preferences and past experiences, generate a personalized trip recommendation with suggested destinations, activities, accommodations, and travel tips."
    return conversation.run(recommendation_prompt)

# Function to refine trip recommendation based on user feedback
def refine_trip_recommendation(feedback):
    refinement_prompt = f"The user is not satisfied with the trip recommendation. They said: '{feedback}'. Generate an alternative trip recommendation that addresses their concerns while still aligning with their preferences and past experiences."
    return conversation.run(refinement_prompt)

# Start Chat Loop with Vacation Preferences Prompt
print("Chatbot is ready! Type 'exit' to stop.")
print("\nI’d love to help you plan your ideal vacation! Let’s explore your travel preferences.\n")
print("1️⃣ Destination Type: Do you prefer beaches, mountains, cities, or countryside escapes?")
print("2️⃣ Activities: Are you into adventure sports, cultural sightseeing, relaxation, or something else?")
print("3️⃣ Accommodation Style: Do you enjoy luxury resorts, boutique hotels, Airbnb stays, or camping?")
print("4️⃣ Climate Preference: Do you love tropical warmth, cool mountain air, or a mix of both?")
print("5️⃣ Food & Dining: Do you like street food, fine dining, or exploring local cuisines?")
print("6️⃣ Travel Pace: Do you prefer a packed itinerary or a laid-back, flexible schedule?")
print("7️⃣ Budget: Do you go for budget-friendly trips, mid-range experiences, or luxury travel?")
print("8️⃣ Company: Do you usually travel solo, with family, friends, or as a couple?")
print("9️⃣ Past Experiences: Can you share specific activities or places from past trips that you enjoyed?")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot session ended.")
        break
    elif user_input.lower() == "recommend trip":
        trip_recommendation = generate_trip_recommendation()
        print(f"Bot: {trip_recommendation}")
        satisfaction = input("Bot: Are you satisfied with this trip recommendation? (yes/no) ")
        if satisfaction.lower() == "no":
            feedback = input("Bot: What part of the trip would you like to change? ")
            alternative_trip = refine_trip_recommendation(feedback)
            print(f"Bot: {alternative_trip}")
    else:
        response = conversation.run(user_input)
        print(f"Bot: {response}")