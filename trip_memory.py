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

class TripMemory:
    def __init__(self):
        # Initialize LLM using langchain + Ollama
        self.llm = ChatOllama(model="llama3.2:latest", temperature=0.7)
        
        # Structure to hold extracted preferences
        self.user_preferences = {
            'activities': {},
            'travel_styles': {},
            'climate_preference': {},
            'budget_level': {}
        }
        
        # Conversation context
        self.messages_history = []
        self.user_name = None
        self.information_collected = {
            'name': False,
            'activities': False,
            'travel_styles': False,
            'climate_preference': False,
            'budget_level': False
        }
        
        self.nlp = spacy.load('en_core_web_sm')
        self.past_trips = []
        self.user_preferences = {
            'activities': {},
            'travel_styles': {},
            'climate_preference': {},
            'budget_level': {}
        }
        self.user_name = None
        
        # Add conversation variety with multiple response options
        self.greetings = [
            "Hi there! I'm your personal Travel Buddy! 🌎",
            "Welcome to your travel planning adventure! ✈️",
            "Hello! Ready to explore the world together? 🗺️"
        ]
        
        self.name_prompts = [
            "I'd love to know your name! 😊",
            "What should I call you? 👋",
            "Who do I have the pleasure of helping today? ✨"
        ]
        
        self.trip_prompts = [
            "Tell me about a memorable trip you've taken! Where did you go? What made it special? 🌟",
            "Share one of your favorite travel experiences with me! 🎒",
            "I'd love to hear about a journey that left an impression on you! 🌅"
        ]
        
        self.encouragements = [
            "That sounds amazing! ✨",
            "How wonderful! 🌟",
            "What a fantastic experience! 🎉",
            "That must have been incredible! 🌈"
        ]

        # Align with TripAdvisor's destination attributes
        self.preference_categories = {
            'activities': set(),
            'travel_styles': set(),
            'climate_preference': set(),
            'budget_level': set()
        }

        self.required_info = {
            'name': False,
            'activities': False,
            'travel_styles': False,
            'climate_preference': False,
            'budget_level': False
        }

        # Add a conversation history to track context
        self.conversation_history = []
        self.information_collected = {
            'name': False,
            'activities': False,
            'travel_styles': False,
            'climate_preference': False,
            'budget_level': False
        }

    def typing_effect(self, text):
        """Simulate typing for a more natural feel"""
        print()
        for char in text:
            print(char, end='', flush=True)
            time.sleep(0.01)
        print()

    def get_valid_input(self, prompt, input_type="text", allow_empty=False):
        while True:
            user_input = input(prompt).strip()
            if not allow_empty and not user_input:
                print("I didn't catch that. Could you please provide an answer?")
                continue
            if input_type == "number":
                try:
                    return float(user_input)
                except ValueError:
                    print("Please enter a valid number.")
            else:
                return user_input

    def analyze_text(self, text, context):
        """
        Analyze text and print extracted information
        """
        doc = self.nlp(text.lower())
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        print(f"\n📝 Analysis of your {context}:")
        sentiment_str = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        print(f"Sentiment: {sentiment_str} ({sentiment:.2f})")

        activities = []
        locations = []
        descriptors = []
        time_refs = []

        print("\nExtracted information:")
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    activity_phrase = ' '.join([t.text for t in token.subtree])
                    activities.append(activity_phrase)
                
                if token.ent_type_ in ["GPE", "LOC"]:
                    locations.append(token.text)
                
                if token.pos_ == "ADJ":
                    descriptors.append(token.text)
                
                if token.ent_type_ == "TIME" or token.ent_type_ == "DATE":
                    time_refs.append(token.text)

        if activities:
            print("\n🎯 Activities detected:")
            for activity in set(activities):
                print(f"- {activity}")
        
        if locations:
            print("\n📍 Locations mentioned:")
            for location in set(locations):
                print(f"- {location}")
        
        if descriptors:
            print("\n✨ Descriptive words used:")
            for desc in set(descriptors):
                print(f"- {desc}")
        
        if time_refs:
            print("\n⏰ Time references:")
            for time in set(time_refs):
                print(f"- {time}")

        self.update_preferences(activities, locations, descriptors, sentiment, context)

    def update_preferences(self, activities, locations, descriptors, sentiment, context):
        for activity, phrase, sent in [(a, a, sentiment) for a in activities]:
            if activity not in self.user_preferences['activities']:
                self.user_preferences['activities'][activity] = 0
            self.user_preferences['activities'][activity] += 1
            if activity not in self.user_preferences['sentiment_by_activity']:
                self.user_preferences['sentiment_by_activity'][activity] = []
            self.user_preferences['sentiment_by_activity'][activity].append(sent)

        pace_indicators = {
            'relaxed': ['relaxing', 'peaceful', 'quiet', 'slow'],
            'busy': ['busy', 'exciting', 'packed', 'full'],
            'adventure': ['adventurous', 'thrilling', 'challenging']
        }

        for descriptor in descriptors:
            for pace, indicators in pace_indicators.items():
                if descriptor in indicators:
                    if pace not in self.user_preferences['travel_pace']:
                        self.user_preferences['travel_pace'][pace] = 0
                    self.user_preferences['travel_pace'][pace] += 1

        if sentiment > 0:
            self.user_preferences['positive_experiences'].append(context)
        elif sentiment < 0:
            self.user_preferences['negative_experiences'].append(context)

    def extract_name(self, text):
        """Use NLP to extract a person's name from text"""
        doc = self.nlp(text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        return names[0] if names else None

    def extract_locations(self, text):
        """Extract location names from text using spaCy's NER"""
        doc = self.nlp(text)
        # Look for GPE (geo-political entities) and LOC (locations)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        return locations

    def extract_dates(self, text):
        """Extract dates from text using spaCy's NER"""
        doc = self.nlp(text)
        # Look for DATE entities
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        return dates

    def get_activity_response(self, activity):
        """Generate personalized responses based on activities"""
        activity_responses = {
            'explore': [
                "Exploring new places is amazing! Did you discover any hidden gems? 💎",
                "Nothing beats the thrill of discovery! What surprised you the most? 🔍"
            ],
            'relax': [
                "Sometimes relaxation is the best part of a trip! How did you unwind? 🌴",
                "Sounds like a perfectly peaceful time! Did you try any local spa treatments? 🧘‍♂️"
            ],
            'adventure': [
                "You're quite the adventurer! What was the most exciting part? 🏃‍♂️",
                "That sounds thrilling! Would you do it again? 🎢"
            ],
            'eat': [
                "Food adventures are the best! What was your favorite local dish? 🍜",
                "Nothing beats local cuisine! Any memorable food moments? 🍽️"
            ],
            'swim': [
                "Beach time is always special! How was the water? 🏊‍♂️",
                "Love a good water adventure! Did you see any marine life? 🐠"
            ],
            'hike': [
                "Hiking is such a rewarding activity! What was the view like? 🏔️",
                "Nothing beats a good hike! Did you reach the summit? 🥾"
            ],
            'shop': [
                "Shopping in new places is so fun! Find any unique souvenirs? 🛍️",
                "Love exploring local markets! What was your best find? 🎁"
            ]
        }
        return random.choice(activity_responses.get(activity, ["Tell me more about that! ✨"]))

    def get_direct_preference(self, category, options):
        """Get explicit preference ratings for a category"""
        self.typing_effect(f"\nLet's talk about your {category.replace('_', ' ')} preferences! 🎯")
        self.typing_effect("Please rate each option from 1-5 (1 = not interested, 5 = love it):")
        
        preferences = {}
        for option in options:
            while True:
                self.typing_effect(f"\nHow do you feel about {option}? (1-5)")
                try:
                    rating = int(input("➡️ "))
                    if 1 <= rating <= 5:
                        preferences[option] = rating
                        break
                    else:
                        self.typing_effect("Please enter a number between 1 and 5! 🎯")
                except ValueError:
                    self.typing_effect("Please enter a valid number! 🎯")
        
        return preferences

    def analyze_trip_text(self, text):
        """Analyze trip description for preferences"""
        doc = self.nlp(text.lower())
        found_preferences = defaultdict(list)
        
        # Activity keywords mapping
        activity_keywords = {
            'explore': ['explore', 'discover', 'sightseeing', 'tour'],
            'swim': ['swim', 'beach', 'snorkel', 'diving'],
            'relax': ['relax', 'chill', 'spa', 'peaceful'],
            'surf': ['surf', 'waves', 'boarding'],
            'eat': ['food', 'eat', 'dining', 'restaurant', 'cuisine'],
            'walk': ['walk', 'stroll', 'wander'],
            'shop': ['shop', 'market', 'mall', 'buying'],
            'drive': ['drive', 'road trip', 'rental car'],
            'adventure': ['adventure', 'exciting', 'thrill'],
            'hike': ['hike', 'trek', 'trail', 'mountain']
        }

        # Style keywords mapping
        style_keywords = {
            'busy': ['busy', 'packed', 'active', 'full'],
            'relaxed': ['relaxed', 'slow', 'peaceful', 'quiet'],
            'cultural': ['culture', 'museum', 'temple', 'history', 'local'],
            'nature': ['nature', 'outdoor', 'wildlife', 'landscape'],
            'adventure': ['adventure', 'exciting', 'thrill', 'extreme']
        }

        # Check for keywords in text
        text_lower = text.lower()
        for category, keywords_dict in [
            ('activities', activity_keywords),
            ('travel_styles', style_keywords)
        ]:
            for pref, keywords in keywords_dict.items():
                if any(keyword in text_lower for keyword in keywords):
                    found_preferences[category].append(pref)

        return found_preferences

    def analyze_user_response(self, text):
        """Use NLP to analyze user response and extract meaningful information"""
        doc = self.nlp(text)
        extracted_info = defaultdict(set)
        
        # Process entities
        for ent in doc.ents:
            # Extract locations
            if ent.label_ in ["GPE", "LOC"]:
                extracted_info['locations'].add(ent.text)
            
            # Extract dates/seasons
            elif ent.label_ == "DATE":
                extracted_info['time_references'].add(ent.text)
            
            # Extract money references for budget
            elif ent.label_ == "MONEY":
                extracted_info['money_references'].add(ent.text)
        
        # Process activities and preferences through verb phrases and context
        for token in doc:
            # Activities often appear as verbs
            if token.pos_ == "VERB":
                verb_phrase = " ".join([child.text for child in token.subtree])
                extracted_info['actions'].add(verb_phrase)
            
            # Find adjectives that might indicate preferences
            if token.pos_ == "ADJ":
                adj_phrase = token.text
                if token.head.pos_ == "NOUN":
                    adj_phrase = f"{token.text} {token.head.text}"
                extracted_info['descriptions'].add(adj_phrase)
        
        # Map extracted information to our preference categories
        self.map_to_preferences(extracted_info, doc.text)
        
        return extracted_info

    def map_to_preferences(self, extracted_info, full_text):
        """Map extracted information to preference categories"""
        
        # Map activities based on verbs and context
        text_lower = full_text.lower()
        
        # Activity detection
        if any(term in text_lower for term in ['explore', 'visit', 'see', 'discover', 'tour']):
            self.preference_categories['activities'].add('explore')
        
        if any(term in text_lower for term in ['swim', 'beach', 'ocean', 'water', 'pool', 'snorkel']):
            self.preference_categories['activities'].add('swim')
            
        if any(term in text_lower for term in ['relax', 'rest', 'chill', 'unwind', 'peaceful', 'quiet']):
            self.preference_categories['activities'].add('relax')
            
        if any(term in text_lower for term in ['eat', 'food', 'restaurant', 'cuisine', 'dining', 'taste']):
            self.preference_categories['activities'].add('eat')
            
        if any(term in text_lower for term in ['walk', 'stroll', 'wander', 'hike', 'trek', 'trail']):
            self.preference_categories['activities'].add('walk' if 'city' in text_lower else 'hike')
            
        if any(term in text_lower for term in ['shop', 'buy', 'purchase', 'store', 'market', 'mall']):
            self.preference_categories['activities'].add('shop')
        
        # Travel style detection
        if any(term in text_lower for term in ['busy', 'packed', 'many', 'lots', 'full', 'active']):
            self.preference_categories['travel_styles'].add('busy')
            
        if any(term in text_lower for term in ['relax', 'slow', 'peaceful', 'easy', 'calm']):
            self.preference_categories['travel_styles'].add('relaxed')
            
        if any(term in text_lower for term in ['culture', 'museum', 'history', 'art', 'local', 'tradition']):
            self.preference_categories['travel_styles'].add('cultural')
            
        if any(term in text_lower for term in ['nature', 'outdoor', 'wild', 'animal', 'park', 'mountains']):
            self.preference_categories['travel_styles'].add('nature')
            
        if any(term in text_lower for term in ['adventure', 'exciting', 'thrill', 'extreme', 'adrenaline']):
            self.preference_categories['travel_styles'].add('adventure')
        
        # Climate preference detection
        if any(term in text_lower for term in ['hot', 'warm', 'tropical', 'beach', 'sun', 'humid']):
            self.preference_categories['climate_preference'].add('tropical')
            
        if any(term in text_lower for term in ['mild', 'spring', 'fall', 'pleasant', 'moderate']):
            self.preference_categories['climate_preference'].add('moderate')
            
        if any(term in text_lower for term in ['cold', 'snow', 'winter', 'cool', 'mountain', 'ski']):
            self.preference_categories['climate_preference'].add('cold')
        
        # Budget level detection
        if any(term in text_lower for term in ['cheap', 'budget', 'affordable', 'inexpensive', 'hostel']):
            self.preference_categories['budget_level'].add('budget')
            
        if any(term in text_lower for term in ['reasonable', 'moderate', 'average', 'mid-range']):
            self.preference_categories['budget_level'].add('moderate')
            
        if any(term in text_lower for term in ['luxury', 'expensive', 'high-end', 'five-star', 'premium']):
            self.preference_categories['budget_level'].add('luxury')

    def generate_follow_up_question(self, missing_categories):
        """Generate a contextual follow-up question based on what information is still needed"""
        category = random.choice(list(missing_categories))
        
        if category == 'activities':
            questions = [
                "What kinds of activities did you enjoy during your travels?",
                "What did you spend most of your time doing on this trip?",
                "What was the most enjoyable activity you participated in?"
            ]
            
        elif category == 'travel_styles':
            questions = [
                "How would you describe your pace of travel? Do you prefer busy itineraries or a more relaxed approach?",
                "Would you say you were more focused on cultural experiences, nature, or something else?",
                "What was the overall style of your trip?"
            ]
            
        elif category == 'climate_preference':
            questions = [
                "How did you feel about the weather during your trip?",
                "Do you typically prefer warm destinations or cooler climates?",
                "What kind of climate did you experience and did you enjoy it?"
            ]
            
        elif category == 'budget_level':
            questions = [
                "How would you describe your spending on this trip?",
                "Did you splurge on luxury experiences or keep things more budget-friendly?",
                "What was your approach to accommodation and dining during this trip?"
            ]
            
        return random.choice(questions)

    def get_name(self):
        """Get user's name through conversation"""
        self.typing_effect("Hi there! I'm your personal travel advisor. What's your name?")
        
        while not self.required_info['name']:
            name_input = input("➡️ ").strip()
            self.conversation_history.append(name_input)
            
            # Try to extract name using NLP
            extracted_name = self.extract_name(name_input)
            if extracted_name:
                user_name = extracted_name
                self.required_info['name'] = True
                self.typing_effect(f"It's great to meet you, {user_name}! I'm here to help you plan your perfect trip.")
                return user_name
            else:
                # If no name detected, ask more directly
                self.typing_effect("I'd like to address you by name. Could you tell me what to call you?")
                direct_name = input("➡️ ").strip()
                if direct_name:
                    self.required_info['name'] = True
                    self.typing_effect(f"Thanks, {direct_name}! Looking forward to helping you plan your travels.")
                    return direct_name

    def collect_past_trips(self):
        """Main method to collect user preferences through conversational interaction"""
        # Track what information we've asked about
        asked_about = {
            'activities': False,
            'travel_styles': False,
            'climate_preference': False,
            'budget_level': False
        }
        
        # Initial prompt setting expectations for concise responses
        system_prompt = """
        You are a travel advisor helping users plan trips. You need to gather their preferences efficiently.
        
        KEEP ALL RESPONSES SHORT AND DIRECT - 2-3 SENTENCES MAXIMUM.
        
        For each response:
        1. Briefly acknowledge what they said (1 short sentence)
        2. Ask ONE specific question about missing information
        
        You need to collect:
        - Activities they enjoy (explore, swim, relax, eat, etc.)
        - Travel style (busy vs relaxed, cultural vs adventure)
        - Climate preferences (tropical, moderate, cold)
        - Budget level (budget, moderate, luxury)
        """
        
        # First message: Ask for user's name
        self.typing_effect("Hi there! I'm your travel advisor. What's your name?")
        
        # Get user's name
        name_input = input("➡️ ").strip()
        
        # Try to extract name
        potential_name = self.extract_name(name_input)
        self.user_name = potential_name if potential_name else name_input
        
        # Start conversation history with this exchange
        self.messages_history = [
            AIMessage(content="Hi there! I'm your travel advisor. What's your name?"),
            HumanMessage(content=name_input)
        ]
        
        # Second message: Ask about past trips
        second_prompt = f"Great to meet you, {self.user_name}! Could you tell me about a memorable trip you've taken?"
        self.typing_effect(second_prompt)
        
        # Update conversation history
        self.messages_history.append(AIMessage(content=second_prompt))
        
        # Set name as collected
        self.information_collected['name'] = True
        
        # Continue conversation until we have all required information
        max_turns = 15
        turns = 0
        
        while not self.check_information_completeness() and turns < max_turns:
            # Get user input
            user_input = input("➡️ ").strip()
            
            # Check for conversation exit
            if user_input.lower() in ["exit", "quit", "bye"]:
                self.typing_effect(f"Thanks for sharing, {self.user_name}! I've noted your preferences.")
                break
                
            # Add to conversation history
            user_message = HumanMessage(content=user_input)
            self.messages_history.append(user_message)
            
            # Extract preferences from user input
            new_preferences = self.extract_preferences(user_input)
            
            if new_preferences:
                self.update_preferences(new_preferences)
            
            # Determine what information is still missing
            missing_categories = []
            for category, collected in self.information_collected.items():
                if not collected and category != 'name':
                    missing_categories.append(category)
            
            # Look at preference categories with less than 2 items
            sparse_categories = []
            for category, prefs in self.user_preferences.items():
                if len(prefs) < 2:
                    sparse_categories.append(category)
            
            # Prioritize categories we haven't specifically asked about yet
            next_category = None
            for category in missing_categories + sparse_categories:
                if not asked_about.get(category, True):
                    next_category = category
                    asked_about[category] = True
                    break
            
            # If we've asked about everything but still missing info, pick randomly
            if next_category is None and missing_categories:
                next_category = missing_categories[0]
            
            # Create focused, direct guidance
            if next_category == 'activities':
                guidance = "Briefly acknowledge their response, then ask DIRECTLY about activities they enjoy during travel."
            elif next_category == 'travel_styles':
                guidance = "Briefly acknowledge their response, then ask DIRECTLY if they prefer busy or relaxed travel pace."
            elif next_category == 'climate_preference':
                guidance = "Briefly acknowledge their response, then ask DIRECTLY about preferred climate (tropical, moderate, cold)."
            elif next_category == 'budget_level':
                guidance = "Briefly acknowledge their response, then ask DIRECTLY about their budget level when traveling."
            else:
                guidance = "Briefly acknowledge their response, then ask a follow-up question about their travel preferences."
            
            # Generate concise response
            guidance_message = SystemMessage(content=f"""
            KEEP YOUR RESPONSE VERY BRIEF (2-3 SENTENCES MAXIMUM).
            
            {guidance}
            
            Do not write lengthy responses. Be conversational but direct.
            Use their name ({self.user_name}) occasionally to personalize.
            """)
            
            temp_messages = self.messages_history.copy()
            temp_messages.insert(0, guidance_message)
            
            ai_response = self.llm.invoke(temp_messages)
            self.typing_effect(ai_response.content)
            self.messages_history.append(AIMessage(content=ai_response.content))
            
            turns += 1
        
        # Final brief summary
        summary_instruction = f"Give a VERY BRIEF summary of {self.user_name}'s travel preferences (2-3 sentences maximum)."
        
        summary_messages = self.messages_history.copy()
        summary_messages.append(HumanMessage(content=summary_instruction))
        
        guidance_message = SystemMessage(content="Keep your summary extremely brief and focused only on their preferences.")
        summary_messages.insert(0, guidance_message)
        
        summary = self.llm.invoke(summary_messages)
        self.typing_effect(summary.content)
        
        return self.user_preferences, self.user_name

    def summarize_preferences(self):
        print(f"\nTravel Profile for {self.user_name}")
        print("=" * 50)

        print("\n📊 Activity Preferences:")
        for activity, count in self.user_preferences['activities'].most_common(5):
            sentiments = self.user_preferences['sentiment_by_activity'].get(activity, [])
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            sentiment_emoji = "❤️" if avg_sentiment > 0.3 else "👍" if avg_sentiment > 0 else "😐"
            print(f"{sentiment_emoji} {activity} (mentioned {count} times)")

        print("\n🏃 Travel Pace:")
        for pace, count in self.user_preferences['travel_pace'].most_common():
            print(f"- {pace}: {count} references")

        print("\n💰 Budget Preferences:")
        budget_counter = Counter(self.user_preferences['budget_pattern'])
        for budget_type, count in budget_counter.most_common():
            print(f"- {budget_type}: {count} trips")

        print("\n🌤️ Seasonal Preferences:")
        for season, count in self.user_preferences['seasonal_preference'].most_common():
            print(f"- {season}: {count} trips")

        if self.user_preferences['positive_experiences']:
            print("\n✨ What You Typically Enjoy:")
            positive_words = [word.text for exp in self.user_preferences['positive_experiences'] 
                            for word in self.nlp(exp) if word.pos_ == "ADJ"]
            for word, count in Counter(positive_words).most_common(3):
                print(f"- {word}")

        if self.user_preferences['negative_experiences']:
            print("\n⚠️ What You Tend to Avoid:")
            negative_words = [word.text for exp in self.user_preferences['negative_experiences'] 
                            for word in self.nlp(exp) if word.pos_ == "ADJ"]
            for word, count in Counter(negative_words).most_common(3):
                print(f"- {word}")

        print("\n💡 Key Insights:")
        comfort_score = sum(1 for exp in self.user_preferences['comfort_preferences'] 
                          if any(word in exp.lower() for word in ['comfortable', 'luxury', 'easy']))
        adventure_score = sum(1 for exp in self.user_preferences['activities'] 
                            if any(word in exp.lower() for word in ['adventure', 'challenge', 'explore']))
        
        if comfort_score > adventure_score:
            print("- You prioritize comfort and convenience in your travels")
        elif adventure_score > comfort_score:
            print("- You're an adventure seeker who enjoys new challenges")
        else:
            print("- You maintain a good balance between comfort and adventure")

    def add_message(self, role, content):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
    def get_llm_response(self, system_prompt=None):
        """Get a response from the LLM using langchain"""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add conversation history
        messages.extend(self.messages_history)
        
        # Get response
        response = self.llm.invoke(messages)
        return response.content
    
    def extract_preferences(self, user_input):
        """Extract travel preferences using the LLM with improved JSON handling"""
        system_prompt = """
        You are an AI assistant with the SOLE purpose of extracting travel preferences from text.
        
        Your output MUST be a valid JSON object and NOTHING ELSE.
        
        Analyze the text and extract preferences in these categories:
        1. Activities (explore, swim, relax, surf, eat, walk, shop, adventure, hike)
        2. Travel styles (busy, relaxed, cultural, nature, adventure)
        3. Climate preference (tropical, moderate, cold)
        4. Budget level (budget, moderate, luxury)
        
        Format your response EXACTLY like this, with no additional text or explanation:
        {
            "activities": {"explore": 0.8, "relax": 0.5},
            "travel_styles": {"busy": 0.3, "cultural": 0.9},
            "climate_preference": {"tropical": 0.7},
            "budget_level": {"moderate": 0.6}
        }
        
        Only include preferences that are mentioned or implied.
        Use a confidence score between 0 and 1 for each preference.
        """
        
        try:
            # Use a separate LLM instance with lower temperature for extraction
            extraction_llm = ChatOllama(model="llama3.2:latest", temperature=0.1)
            
            # Create a structured extraction prompt
            extraction_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract travel preferences from this text. Remember to output ONLY valid JSON:\n\n{user_input}")
            ]
            
            # Get the extraction result
            extraction_result = extraction_llm.invoke(extraction_messages)
            response_text = extraction_result.content
            
            # Debug: print the raw response
            print("\n--- Raw LLM Response ---")
            print(response_text)
            print("------------------------\n")
            
            # Try different JSON extraction methods
            
            # Method 1: Direct parsing if it's clean JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # Method 2: Find JSON between curly braces
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Method 3: Use regex to find most JSON-like content
            import re
            json_pattern = r'\{(?:[^{}]|(?R))*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    pass
            
            print("All JSON parsing methods failed, using manual extraction")
            
            # Method 4: Manual extraction of preferences
            preferences = {
                "activities": {},
                "travel_styles": {},
                "climate_preference": {},
                "budget_level": {}
            }
            
            # Define potential keywords for each category
            keywords = {
                "activities": ["explore", "swim", "relax", "surf", "eat", "walk", "shop", "adventure", "hike"],
                "travel_styles": ["busy", "relaxed", "cultural", "nature", "adventure"],
                "climate_preference": ["tropical", "moderate", "cold"],
                "budget_level": ["budget", "moderate", "luxury"]
            }
            
            # Look for mentions of each keyword in the response text
            for category, words in keywords.items():
                for word in words:
                    if word.lower() in user_input.lower():
                        preferences[category][word] = 0.7  # Default confidence
            
            return preferences
            
        except Exception as e:
            print(f"Error in preference extraction: {e}")
            
            # Return a default structure with empty preferences
            return {
                "activities": {},
                "travel_styles": {},
                "climate_preference": {},
                "budget_level": {}
            }
    
    def update_preferences(self, new_preferences):
        """Update user preferences with new information, handling null values"""
        for category, prefs in new_preferences.items():
            # Skip if this category doesn't exist in our preference structure
            if category not in self.user_preferences:
                continue
            
            # Skip null/None values
            if prefs is None:
                continue
            
            # Now we know prefs is a dictionary, so we can iterate through items
            for pref, score in prefs.items():
                # Convert score to float if it's not already
                try:
                    score_value = float(score)
                except (ValueError, TypeError):
                    # If conversion fails, use a default value
                    score_value = 0.5
                
                # Update the preference score
                if pref in self.user_preferences[category]:
                    # Take the higher confidence score
                    self.user_preferences[category][pref] = max(
                        self.user_preferences[category][pref], score_value
                    )
                else:
                    self.user_preferences[category][pref] = score_value

    def check_information_completeness(self):
        """Check if we have collected sufficient information"""
        for category, prefs in self.user_preferences.items():
            if prefs and len(prefs) >= 2:
                self.information_collected[category] = True
        
        if self.user_name:
            self.information_collected['name'] = True
            
        return all(self.information_collected.values())
    
    def extract_name(self, response):
        """Extract user's name from conversation"""
        system_prompt = "Extract the user's name from this text if present. Return ONLY the name with no additional text. If no name is found, return 'None'."
        
        extraction_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract the name from this text: {response}")
        ]
        
        result = self.llm.invoke(extraction_messages)
        name = result.content.strip()
        
        return None if name == "None" else name
