import streamlit as st
from travelAgent import run_conversation, list_existing_users
import os
from tts import TextToSpeech

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'tts' not in st.session_state:
        st.session_state.tts = TextToSpeech()

def main():
    st.title("AI Travel Agent 🌎✈️")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for user management
    with st.sidebar:
        st.header("User Management")
        
        # Option to enter existing user ID
        existing_id = st.text_input("Enter your user ID (if returning):")
        if existing_id:
            if os.path.exists(os.path.join("./travel_memory", existing_id)):
                import travelAgent
                travelAgent.is_new_user = False
                st.session_state.user_id = existing_id
                st.success(f"Welcome back! User ID: {existing_id}")
            else:
                st.error("User ID not found.")
        
        # Option to create new user
        if st.button("Start New Session"):
            import travelAgent
            travelAgent.is_new_user = True
            _, response = run_conversation("Hello", None)  # This will create a new user
            st.session_state.user_id = _
            st.success(f"New session created! Your User ID: {st.session_state.user_id}")
            # Play welcome message
            st.session_state.tts.play(response)
        
        # Show existing users
        st.subheader("Existing Users")
        try:
            base_dir = "./travel_memory"
            if os.path.exists(base_dir):
                users = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                if users:
                    st.write("Available user IDs:")
                    for user in users:
                        st.code(user)
                else:
                    st.write("No existing users found.")
        except Exception as e:
            st.error(f"Error listing users: {e}")
    
    # Main chat interface
    st.header("Chat with your Travel Agent")
    
    # Display current user ID
    if st.session_state.user_id:
        st.info(f"Current User ID: {st.session_state.user_id}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What's your travel question?"):
        if not st.session_state.user_id:
            st.error("Please start a new session or enter an existing user ID first.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                _, response = run_conversation(prompt, st.session_state.user_id)
                st.markdown(response)
                # Play the response audio
                st.session_state.tts.play(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 