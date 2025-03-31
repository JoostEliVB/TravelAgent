# DSAIT4065_TravelAgent

very basic agent is available in testing.ipynb. Short-term memory present with token-trimmer.
Using Ollama llama3:latest model for llm, LangChain framework for agent interaction and (eventually) LangGraph for memory framework.

Download Ollama and run command `ollama pull llama3.2` to download the llm (4.7GB)

reqs:

```
pip install langchain-core langgraph>0.2.27
pip install -U langchain-ollama
```

## Setup Text-To-Speech

1. Get your Google Cloud API credentials:
   - Go to Google Cloud Console
   - Create a service account
   - Download the JSON key file
2. Rename the key file to `API_key.json`
3. Place it in the project root directory
