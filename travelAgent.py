from typing import Dict, List, Tuple, Any
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langgraph.graph import Graph, StateGraph
from pydantic import BaseModel
import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
import uuid
import random

# Load environment variables
load_dotenv()

# Get and validate API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize OpenAI Chat model with GPT-4o-mini
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.7,
    max_tokens=2000
)

# Initialize embeddings with OpenAI
embeddings = OpenAIEmbeddings(
    api_key=api_key
)

# Define the state schema
class AgentState(BaseModel):
    user_id: str
    messages: List[Dict[str, str]]
    memory: List[Dict[str, str]]
    current_user_input: str = ""
    extracted_info: Dict[str, Any] = {}
    last_recommendation: str = ""
    user_profile: Dict[str, Any] = {}
    chat_history: List[Tuple[str, str]] = []

    @classmethod
    def create_new_user(cls, custom_id: str = None) -> 'AgentState':
        """Create a new AgentState instance with a unique three-digit user ID."""
        def generate_unique_id():
            """Generate a unique three-digit ID that doesn't exist in the travel_memory directory."""
            while True:
                # Generate a random three-digit number
                user_id = str(random.randint(100, 999))
                # Check if this ID already exists
                user_dir = os.path.join("./travel_memory", user_id)
                if not os.path.exists(user_dir):
                    return user_id
        
        # Use custom ID if provided, otherwise generate a new one
        user_id = custom_id if custom_id else generate_unique_id()
        
        return cls(
            user_id=user_id,
            messages=[],
            memory=[],
            current_user_input="",
            extracted_info={},
            last_recommendation="",
            user_profile={},
            chat_history=[]
        )

# Memory system prompts
MEMORY_CHECK_PROMPT = """You are a memory system for a travel agent. Your task is to determine if the user's input contains new information that should be memorized.
Consider the following types of information:
- User's name or personal identifiers
- Vacation preferences (e.g., beach vs. city, luxury vs. budget)
- Past vacation experiences (destinations, activities, duration)
- Location information (where they live, where they've been)
- Travel style preferences (solo, family, group)
- Budget information (price range, willingness to spend)
- Family composition (number of children, ages)
- Time preferences (favorite seasons, preferred duration)
- Travel frequency
- Special requirements (accessibility, dietary restrictions)
- Bucket list destinations

User input: {input}

Does this input contain new information that should be memorized? Answer with 'yes' or 'no' only."""

INFO_EXTRACTION_PROMPT = """Extract the relevant information from the user's input and format it as a simple JSON object.
Focus on:
- User's name (if mentioned in any way, including introductions like "I'm [name]" or "My name is [name]")
- Vacation preferences
- Past vacation experiences
- Location information
- Travel style preferences
- Budget information
- Family composition
- Time preferences
- Travel frequency
- Special requirements
- Bucket list destinations

User input: {input}

Return ONLY a valid JSON object without any additional text or formatting. 
If a name is mentioned, include it as "user_name" in the JSON.

Examples:
{{"user_name": "John", "timestamp": "2024-04-01"}}
{{"vacation_preferences": {{"type": "beach"}}, "timestamp": "2024-04-01"}}
{{"user_name": "Sarah", "vacation_preferences": {{"type": "mountain"}}, "timestamp": "2024-04-01"}}"""

# Main agent prompt
AGENT_PROMPT = """You are an experienced travel agent with access to the user's preferences and past experiences.
Your goal is to provide personalized travel recommendations and engage in meaningful conversations about travel experiences.

User Profile:
{user_profile}

Recent Memories:
{memory}

Previous conversation:
{chat_history}

Last Recommendation:
{last_recommendation}

Guidelines:
1. Be conversational and friendly while maintaining professionalism
2. Keep ALL responses to exactly TWO sentences maximum
3. Provide specific recommendations based on the user's profile
4. If recommending a destination, explain why it matches their preferences
5. Consider their past experiences when suggesting new destinations
6. Be mindful of their budget and travel style
7. If they mention dislikes or negative experiences, avoid similar suggestions
8. Use your extensive knowledge of destinations, cultures, and travel logistics
9. Provide practical advice about timing, logistics, and local customs
10. NEVER recommend more than two destinations

User: {input}
Travel Agent:"""

# Recommendation generation prompt
RECOMMENDATION_PROMPT = """Based on the user's profile and preferences, generate a personalized travel recommendation in a natural, conversational style.

User Profile:
{user_profile}

Recent Memories:
{memory}

Previous Recommendations:
{last_recommendation}

IMPORTANT: Only provide a recommendation if the user has specified their budget, number of companions, and preferred travel time. If any of these details are missing, ask the user for the missing information instead.

IMPORTANT: You MUST limit your recommendation to exactly TWO destinations

STRICT RULES:
- Your response MUST be EXACTLY TWO sentences
- First sentence: Recommend your primary destination and explain why it's perfect for them
- Second sentence: Mention the best time to visit and one unique highlight
- Third sentence: Briefly suggest ONE alternative destination that matches their preferences

Write your response as if you're having a natural conversation with the user. Weave the information into a flowing narrative that highlights why these destinations would be perfect for them based on their preferences and past experiences."""

# Initialize prompts
memory_check_prompt = PromptTemplate(
    input_variables=["input"],
    template=MEMORY_CHECK_PROMPT
)

info_extraction_prompt = PromptTemplate(
    input_variables=["input"],
    template=INFO_EXTRACTION_PROMPT
)

agent_prompt = PromptTemplate(
    input_variables=["memory", "chat_history", "input", "user_profile", "last_recommendation"],
    template=AGENT_PROMPT
)

recommendation_prompt = PromptTemplate(
    input_variables=["user_profile", "memory", "last_recommendation"],
    template=RECOMMENDATION_PROMPT
)

# Initialize chains
memory_check_chain = LLMChain(llm=llm, prompt=memory_check_prompt)
info_extraction_chain = LLMChain(llm=llm, prompt=info_extraction_prompt)
agent_chain = LLMChain(llm=llm, prompt=agent_prompt)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Initialize vector store for persistent memory with user-specific collections
class UserAwareChroma:
    def __init__(self, base_dir: str, embedding_function):
        self.base_dir = base_dir
        self.embedding_function = embedding_function
        self.stores: Dict[str, Chroma] = {}

    def get_store(self, user_id: str) -> Chroma:
        """Get or create a Chroma store for a specific user."""
        if user_id not in self.stores:
            user_dir = os.path.join(self.base_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            self.stores[user_id] = Chroma(
                persist_directory=user_dir,
                embedding_function=self.embedding_function
            )
        return self.stores[user_id]

# Initialize the user-aware vector store
vectorstore = UserAwareChroma(
    base_dir="./travel_memory",
    embedding_function=OpenAIEmbeddings(api_key=api_key)
)

# Dictionary to store user-specific memory buffers
user_memories: Dict[str, ConversationBufferMemory] = {}

def get_user_memory(user_id: str) -> ConversationBufferMemory:
    """Get or create a memory buffer for a specific user."""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(
            k=2,  # Keep only the last 2 interactions
            return_messages=True,
            memory_key="chat_history"
        )
    return user_memories[user_id]

def update_user_profile(state: AgentState, new_info: Dict[str, Any]) -> AgentState:
    """Update the user profile with new information."""
    try:
        for key, value in new_info.items():
            if key != "timestamp":
                if key not in state.user_profile:
                    state.user_profile[key] = []
                state.user_profile[key].append({
                    "value": value,
                    "timestamp": new_info.get("timestamp", datetime.now().isoformat())
                })
        print(f"Updated user profile with new information")
    except Exception as e:
        print(f"Error updating user profile: {str(e)}")
    return state

def check_for_new_info(state: AgentState) -> AgentState:
    """Check if the user input contains new information to memorize."""
    try:
        result = memory_check_chain.invoke({"input": state.current_user_input})
        if result["text"].strip().lower() == 'yes':
            # Extract information
            extracted = info_extraction_chain.invoke({"input": state.current_user_input})
            try:
                # Get the raw text response
                extracted_text = extracted["text"].strip()
                
                # Clean up any potential markdown or extra formatting
                if "```" in extracted_text:
                    # Extract content between triple backticks if present
                    start = extracted_text.find("{")
                    end = extracted_text.rfind("}") + 1
                    if start != -1 and end != -1:
                        extracted_text = extracted_text[start:end]
                
                # Try to parse the JSON
                extracted_info = json.loads(extracted_text)
                
                if not isinstance(extracted_info, dict):
                    print("Error: Extracted information is not a dictionary")
                    return state
                
                # Add timestamp if not present
                if "timestamp" not in extracted_info:
                    extracted_info["timestamp"] = datetime.now().isoformat()
                
                state.extracted_info = extracted_info
                
                # Update user profile
                state = update_user_profile(state, extracted_info)
                
                # Store in user-specific vector store
                user_store = vectorstore.get_store(state.user_id)
                for key, value in extracted_info.items():
                    if key != "timestamp":  # Don't store timestamp as separate memory
                        memory_text = f"{key}: {json.dumps(value)}"
                        user_store.add_texts(
                            texts=[memory_text],
                            metadatas=[{
                                "type": key,
                                "timestamp": extracted_info["timestamp"],
                                "user_id": state.user_id
                            }]
                        )
                        print(f"Stored new information: {memory_text}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Problematic text: {extracted_text}")
            except Exception as e:
                print(f"Error processing information: {str(e)}")
    except Exception as e:
        print(f"Error in memory check: {str(e)}")
    
    return state

def has_essential_travel_info(user_profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if the user profile has all essential travel information.
    
    Returns:
        Tuple of (has_all_info: bool, missing_info: List[str])
    """
    required_fields = {
        "budget": "budget information",
        "travel_companions": "number of companions",
        "travel_time": "preferred travel time"
    }
    
    missing_info = []
    for field, description in required_fields.items():
        if field not in user_profile or not user_profile[field]:
            missing_info.append(description)
    
    return len(missing_info) == 0, missing_info

def generate_response(state: AgentState) -> AgentState:
    """Generate a response using the agent chain."""
    # Get user-specific vector store and memory
    user_store = vectorstore.get_store(state.user_id)
    user_memory = get_user_memory(state.user_id)
    
    # Check if this is a new user by checking for existing memories
    collection = user_store._collection
    total_docs = len(collection.get()['ids'])

    
    # Get chat history from user-specific memory buffer
    chat_history = user_memory.load_memory_variables({})["chat_history"]
    
    # Get user profile
    user_profile = state.user_profile
    user_profile_text = "\n".join([f"{k}: {v}" for k, v in user_profile.items()])  

    # Only use memories for existing users
    if not is_new_user:
        # Set k to min of total docs or 3
        k = min(3, max(1, total_docs))
        
        # Retrieve relevant memories for this user
        relevant_memories = user_store.similarity_search(
            state.current_user_input,
            k=k
        )
        memory_text = "\n".join([doc.page_content for doc in relevant_memories]) if relevant_memories else ""
    else:
        memory_text = ""
        # For new users, only keep last five  messages in chat history
        if len(chat_history) > 5:
            chat_history = chat_history[-5:]
        print("Note: New user session - only storing new memories")
    
    # Generate response using the agent chain
    response = agent_chain.invoke({
        "memory": memory_text,
        "chat_history": chat_history,
        "input": state.current_user_input,
        "user_profile": user_profile_text,
        "last_recommendation": state.last_recommendation
    })
    
    # Extract the text from the response
    response_text = response["text"]
    
    # Update user-specific memory with the new interaction
    user_memory.save_context(
        {"input": state.current_user_input},
        {"output": response_text}
    )
    
    # Update state with the response
    state.messages.append({"role": "user", "content": state.current_user_input})
    state.messages.append({"role": "assistant", "content": response_text})
    state.last_recommendation = response_text
    

    return state

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("check_memory", check_for_new_info)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("check_memory", "generate_response")

# Set entry point
workflow.set_entry_point("check_memory")

# Compile the graph
app = workflow.compile()

# Dictionary to store active user sessions
active_sessions: Dict[str, AgentState] = {}

# Global flag for new user status
is_new_user = True

def run_conversation(user_input: str, user_id: str = None) -> Tuple[str, str]:
    """Run a single turn of conversation for a specific user."""
    global is_new_user  # Access the global variable
    
    # Get or create user session
    if user_id and user_id in active_sessions:
        state = active_sessions[user_id]
    else:
        # Create new user session and clear any existing memory
        state = AgentState.create_new_user(custom_id=user_id)
        user_id = state.user_id
        active_sessions[user_id] = state
        # Initialize fresh memory for new user
        user_memories[user_id] = ConversationBufferMemory(
            k=2,
            return_messages=True,
            memory_key="chat_history"
        )
    
    state.current_user_input = user_input
    result = app.invoke(state)
    
    # Update session state
    active_sessions[user_id] = AgentState(
        user_id=user_id,
        messages=result.get("messages", state.messages),
        memory=result.get("memory", state.memory),
        current_user_input=result.get("current_user_input", state.current_user_input),
        extracted_info=result.get("extracted_info", state.extracted_info),
        last_recommendation=result.get("last_recommendation", state.last_recommendation),
        user_profile=result.get("user_profile", state.user_profile)
    )
    
    # Get the last assistant message
    last_message = next((msg["content"] for msg in reversed(result.get("messages", [])) 
                        if msg["role"] == "assistant"), 
                       "I apologize, but I couldn't generate a response.")
    
    return user_id, last_message

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Travel Agent Chatbot')
    parser.add_argument('--user-id', type=str, help='Existing user ID to load a specific user session')
    parser.add_argument('--new-user-id', type=str, help='Custom three-digit ID for a new user session')
    parser.add_argument('--list-users', action='store_true', help='List all existing user IDs')
    return parser.parse_args()

def validate_user_id(user_id: str) -> bool:
    """Validate if a user ID is a valid three-digit number."""
    try:
        num = int(user_id)
        return 100 <= num <= 999
    except ValueError:
        return False

def list_existing_users():
    """List all existing user IDs from the travel_memory directory."""
    try:
        base_dir = "./travel_memory"
        if not os.path.exists(base_dir):
            print("No existing users found.")
            return
        
        users = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not users:
            print("No existing users found.")
            return
        
        print("\nExisting user IDs:")
        for user_id in users:
            print(f"- {user_id}")
    except Exception as e:
        print(f"Error listing users: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle --list-users flag
    if args.list_users:
        list_existing_users()
        sys.exit(0)
    
    print("Welcome to your AI Travel Agent! I'm here to help you plan your next adventure.")
    print("You can ask me for recommendations, share your travel experiences, or discuss your preferences.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    
    # Initialize with provided user ID or create new session
    current_user_id = args.user_id
    if current_user_id:
        print(f"\nLoading existing user session: {current_user_id}")
        # Verify user exists
        user_dir = os.path.join("./travel_memory", current_user_id)
        if not os.path.exists(user_dir):
            print(f"Warning: User ID {current_user_id} not found. Creating new session.")
            current_user_id = None
    else:
        print("\nStarting new user session")
        is_new_user = True
    
    # Handle custom new user ID if provided
    if args.new_user_id:
        if not validate_user_id(args.new_user_id):
            print("Error: New user ID must be a three-digit number (100-999)")
            sys.exit(1)
        user_dir = os.path.join("./travel_memory", args.new_user_id)
        if os.path.exists(user_dir):
            print(f"Error: User ID {args.new_user_id} already exists")
            sys.exit(1)
        current_user_id = args.new_user_id
        print(f"\nCreating new user session with ID: {current_user_id}")
    
    # Start conversation
    print("\nHow can I help you today?")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for chatting with me! Have a great day!")
                print(f"Your user ID is: {current_user_id}")
                print("You can use this ID to continue our conversation later with:")
                print(f"python travelAgent.py --user-id {current_user_id}")
                break
                
            current_user_id, response = run_conversation(user_input, current_user_id)
            print(f"\nTravel Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nConversation interrupted.")
            print(f"Your user ID is: {current_user_id}")
            print("You can use this ID to continue our conversation later with:")
            print(f"python travelAgent.py --user-id {current_user_id}")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")