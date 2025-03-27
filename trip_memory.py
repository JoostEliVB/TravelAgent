from collections import Counter
from datetime import datetime
import spacy
from textblob import TextBlob
import random
import time
from collections import defaultdict
import json
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, List, Any
from database import TravelDatabase

class TravelAgent:
    def __init__(self):
        # Initialize conversation LLM (llama) for basic interaction
        self.conversation_llm = ChatOllama(
            model="llama3.2",
            temperature=0.7,
            system="""You are a travel assistant. Extract travel details from user input.
            Return ONLY a JSON with time_period, companions, and budget (or null if not mentioned)."""
        )
        
        # Initialize recommendation LLM (deepseek) for suggestions
        self.recommendation_llm = ChatOllama(
            model="deepseek-r1:7b",
            temperature=0.8,
            system="You are a travel advisor. Respond with destination recommendations in format: CityName, Country: One-line description"
        )
        
        # Store trip details
        self.trip_details = {
            'time_period': None,
            'companions': None,
            'budget': None
        }

    def start_conversation(self):
        """Start the conversation and collect all information"""
        # Initial greeting and name collection
        self.typing_effect("Hi there! 👋 I'm your friendly travel companion, and I'd love to help you plan your perfect getaway. What should I call you?")
        
        user_response = input("➡️ ").strip()
        name = self.extract_name(user_response)
        
        # Collect basic trip details
        self.typing_effect(f"""Lovely to meet you, {name}! 🌟 
Let's plan your perfect trip. Please tell me:
• When you'd like to travel
• Who you'll be traveling with
• What's your budget (low/medium/high)""")

        self.collect_trip_details()
        self.generate_initial_recommendation()
        self.collect_preferences()
        self.generate_personalized_recommendation()

    def collect_trip_details(self):
        """Collect all necessary trip details"""
        while not all(self.trip_details.values()):
            response = input("➡️ ").strip()
            
            # Extract details from response
            new_details = self.extract_details(response)
            
            # Update trip details
            for key, value in new_details.items():
                if value and value.lower() != 'null':
                    self.trip_details[key] = value
            
            # Ask for missing details
            missing = [k for k, v in self.trip_details.items() if not v]
            if missing:
                self.ask_missing_details(missing)
        
        # Show final summary
        self.show_summary()

    def extract_details(self, user_input: str) -> dict:
        """Extract travel details from user input"""
        prompt = f"""Extract travel details from: '{user_input}'
        Return ONLY a JSON object:
        {{
            "time_period": "when they want to travel (or null if unclear)",
            "companions": "who they're traveling with (or null if unclear)",
            "budget": "their budget level (or null if unclear)"
        }}"""
        
        try:
            response = self.conversation_llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Clean up the response
            if content.startswith('```') and content.endswith('```'):
                content = content[3:-3]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()
            
            return json.loads(content)
            
        except Exception:
            return {'time_period': None, 'companions': None, 'budget': None}

    def ask_missing_details(self, missing: list):
        """Ask for specific missing details"""
        prompts = {
            'time_period': "When would you like to travel?",
            'companions': "Who will you be traveling with?",
            'budget': "What's your budget (low/medium/high)?"
        }
        
        if len(missing) == 1:
            self.typing_effect(f"I just need to know: {prompts[missing[0]]}")
        else:
            self.typing_effect("I still need to know:")
            for m in missing:
                self.typing_effect(f"• {prompts[m]}")

    def show_summary(self):
        """Show summary of collected details"""
        self.typing_effect("\nGreat! Here's what I understand about your trip:")
        self.typing_effect(f"🗓️ When: {self.trip_details['time_period']}")
        self.typing_effect(f"👥 With: {self.trip_details['companions']}")
        self.typing_effect(f"💰 Budget: {self.trip_details['budget']}")

    def generate_initial_recommendation(self):
        """Generate initial travel recommendation using DeepSeek"""
        self.typing_effect("\n🤔 Based on your trip details, let me suggest a destination...")
        
        prompt = f"""Recommend ONE destination for:
        - Time: {self.trip_details['time_period']}
        - Group: {self.trip_details['companions']}
        - Budget: {self.trip_details['budget']}
        
        Response format: CityName, Country: One-line description"""
        
        try:
            response = self.recommendation_llm.invoke([HumanMessage(content=prompt)])
            self.typing_effect(f"\n✨ {response.content.strip()}")
            self.typing_effect("\nWould you like to tell me about your past trips so I can give you more personalized suggestions?")
        except Exception:
            self.typing_effect("\n✨ Barcelona, Spain: Vibrant city with perfect blend of culture, cuisine, and beaches.")
            self.typing_effect("\nWould you like to tell me about your past trips so I can give you more personalized suggestions?")

    def collect_preferences(self):
        """Collect user preferences through natural conversation"""
        self.typing_effect("\nWould you like to tell me about your past trips so I can give you more personalized suggestions?")
        response = input("➡️ ").strip()
        
        if response.lower() in ['no', 'nope', 'n']:
            return
        
        # First trip conversation
        self.typing_effect("Tell me about a memorable trip you've taken - where did you go?")
        self.have_trip_conversation()
        
        # Second trip conversation
        self.typing_effect("\nHow about another trip? Where else have you traveled that you really enjoyed?")
        self.have_trip_conversation()

    def have_trip_conversation(self):
        """Have a natural conversation about a past trip"""
        destination = input("➡️ ").strip()
        
        # Generate follow-up questions based on the destination
        questions = self.generate_follow_up_questions(destination)
        preferences = {}
        
        for question in questions:
            self.typing_effect(question)
            response = input("➡️ ").strip()
            
            # Analyze response and update preferences
            new_prefs = self.analyze_response(response)
            self.update_preferences(new_prefs)

    def generate_follow_up_questions(self, destination: str) -> list:
        """Generate contextual follow-up questions"""
        prompt = f"""Generate 3 natural follow-up questions about their trip to {destination}.
        
        Focus on understanding:
        1. Activities they enjoyed
        2. Their travel style
        3. Weather/timing preferences
        
        Return ONLY the questions, one per line, no numbering."""
        
        try:
            response = self.conversation_llm.invoke([
                SystemMessage(content="You are a friendly travel agent having a natural conversation. Ask engaging follow-up questions."),
                HumanMessage(content=prompt)
            ])
            
            questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            return questions[:3]  # Limit to 3 questions
        except Exception:
            # Fallback questions
            return [
                f"What did you enjoy most about {destination}?",
                "What kind of activities did you do there?",
                "When did you visit, and how was the weather?"
            ]

    def analyze_response(self, response: str) -> dict:
        """Analyze a single response for preferences"""
        analysis_prompt = f"""Analyze this response: '{response}'
        
        Extract preferences about:
        1. Activities they enjoy
        2. Climate/weather they prefer
        3. Travel style
        
        Return ONLY a JSON object:
        {{
            "activities": ["activity1", "activity2"],
            "climate": ["climate1", "climate2"],
            "travel_style": ["style1", "style2"]
        }}"""
        
        try:
            response = self.conversation_llm.invoke([
                SystemMessage(content="You are a preference analyzer. Extract specific preferences from the response."),
                HumanMessage(content=analysis_prompt)
            ])
            
            content = response.content.strip()
            if content.startswith('```') and content.endswith('```'):
                content = content[3:-3]
            if content.startswith('json'):
                content = content[4:]
            
            return json.loads(content.strip())
        except Exception:
            return {'activities': [], 'climate': [], 'travel_style': []}

    def generate_personalized_recommendation(self):
        """Generate personalized recommendation based on collected preferences"""
        if not any(self.user_preferences.values()):
            return
        
        self.typing_effect("\nThanks for sharing your travel experiences! Based on what you've told me, I think you might enjoy this destination...")
        
        prompt = f"""Recommend ONE destination considering:
        Trip Details:
        - Time: {self.trip_details['time_period']}
        - Group: {self.trip_details['companions']}
        - Budget: {self.trip_details['budget']}
        
        User Preferences:
        - Favorite Activities: {', '.join(self.user_preferences['activities'])}
        - Preferred Climate: {', '.join(self.user_preferences['climate'])}
        - Travel Style: {', '.join(self.user_preferences['travel_style'])}
        
        Response format: CityName, Country: One-line description"""
        
        try:
            response = self.recommendation_llm.invoke([HumanMessage(content=prompt)])
            self.typing_effect(f"\n✨ {response.content.strip()}")
        except Exception:
            self.typing_effect("\n✨ Kyoto, Japan: Cultural destination with perfect blend of traditional experiences and modern comfort.")

    def typing_effect(self, text):
        """Print with typing effect"""
        print()
        for char in text:
            print(char, end='', flush=True)
            time.sleep(0.01)
        print()

    def extract_name(self, user_input: str) -> str:
        """Extract user's name using Llama"""
        name_prompt = f"""Extract ONLY the person's name from: '{user_input}'
        
        Rules:
        1. Return ONLY the name, nothing else
        2. If no clear name found, return exactly "friend"
        3. Ignore greetings like "hi", "hello"
        4. Return single word name only
        
        Examples:
        Input: "Hi I'm John" -> "John"
        Input: "Hello" -> "friend"
        Input: "Call me Sarah please" -> "Sarah"
        Input: "My name is Mike Smith" -> "Mike"
        """
        
        try:
            response = self.conversation_llm.invoke([
                SystemMessage(content="You are a name extractor. Return ONLY the name or 'friend'. No other text or punctuation."),
                HumanMessage(content=name_prompt)
            ])
            
            name = response.content.strip()
            
            # Basic validation
            if not name or len(name) > 20:
                return "friend"
            
            return name
            
        except Exception:
            return "friend"

# Run the agent
if __name__ == "__main__":
    agent = TravelAgent()
    agent.start_conversation()
