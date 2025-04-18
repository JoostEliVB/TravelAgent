# Travel Agent with Memory and Voice Interface

A conversational travel agent that uses LangChain and LangGraph to provide personalized travel recommendations. The agent features persistent memory, voice input/output capabilities, and a user-friendly Streamlit interface.

## Features

- **Conversational Interface**: Natural language interaction with the travel agent
- **Persistent Memory**: Maintains conversation history and user preferences across sessions
- **Voice Interface**: 
  - Speech-to-text input for user queries
  - Text-to-speech output for agent responses
- **Personalized Recommendations**: Tailored travel suggestions based on user preferences and history
- **Essential Travel Information**: Tracks and ensures all necessary travel details are collected
- **Multi-User Support**: Separate memory and preferences for different users

## Prerequisites

- Python 3.8 or higher
- Google Cloud account with Text-to-Speech API enabled
- Google Cloud credentials (JSON key file)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following:
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

## Project Structure

- `app.py`: Main Streamlit application interface
- `travelAgent.py`: Core travel agent implementation with LangChain
- `tts.py`: Text-to-speech functionality
- `memory.py`: Memory management system
- `requirements.txt`: Project dependencies

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface through your browser (typically at http://localhost:8501)

3. Interact with the travel agent:
   - Type your queries in the text input
   - Use the "Speak to the Agent" button for voice input
   - Receive both text and voice responses
   - Get personalized travel recommendations

## Features in Detail

### Memory System
- **Short-term Memory**: Recent conversation context
- **Long-term Memory**: Persistent storage of user preferences and history
- **New User Handling**: Fresh start for new users without accessing previous memories

### Travel Recommendations
- Limited to 2 locations per recommendation
- Requires essential travel information:
  - Travel dates
  - Budget
  - Travel style
  - Destination preferences

### Voice Interface
- Speech recognition for user input
- Text-to-speech for agent responses
- Automatic audio management (stops previous audio before playing new responses)

## Dependencies

- langchain>=0.1.0
- langgraph>=0.0.10
- chromadb>=0.4.22
- python-dotenv>=1.0.0
- pydantic>=2.5.2
- google-cloud-texttospeech>=2.14.1
- playsound>=1.3.0
- pygame>=2.5.2

## Contributing

Feel free to submit issues and enhancement requests!

## Privacy

Be aware that the agent makes use of OpenAI services and Google services. Information entered into the travel agent could thus be collected by OpenAI or Google.