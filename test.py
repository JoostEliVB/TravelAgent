# from dotenv import load_dotenv
# from langchain.memory import VectorStoreRetrieverMemory
# from langchain.chains import ConversationChain
# from langchain.prompts import PromptTemplate
# from langchain_milvus.vectorstores import Milvus
#
# from langchain_community.llms import HuggingFacePipeline
# from langchain_community.embeddings import HuggingFaceEmbeddings
#
# # Load Hugging Face Embeddings
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
#
#
# vectordb = Milvus(
#    embeddings,
#    connection_args={"uri": "./milvus_demo.db"},
# )
# retriever = vectordb.as_retriever(search_kwargs=dict(k=1))
# memory = VectorStoreRetrieverMemory(retriever=retriever)
#
#
# about_me = [
#    {"input": "My name is Bob.", "output": "Got it!"},
#    {"input": "I'm from San Francisco.", "output": "Got it!"},
# ]
# for example in about_me:
#    memory.save_context({"input": example["input"]}, {"output": example["output"]})
#
#
# prompt_template = """The following is a friendly conversation between a user and a chatbot. The chatbot is talkative and provides lots of specific details from its context. If the chatbot does not know the answer to a question, it truthfully says it does not know.
#
#
# Relevant pieces of previous conversation:
# {history}
#
#
# (You do not need to use these pieces of information if not relevant)
#
#
# Current conversation:
# User: {input}
# Chatbot:"""
#
#
# prompt = PromptTemplate(input_variables=["history", "input"], template=prompt_template)
#
#
# conversation_with_memory = ConversationChain(
#    llm=llm, prompt=prompt, memory=memory, verbose=True
# )
#
#
# answer = conversation_with_memory.predict(input="What's my name?")
#
# print(answer)