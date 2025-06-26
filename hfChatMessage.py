import os
from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationalChain

HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

#ChatMessageHistory
history = ChatMessageHistory()
history.add_user_message("Hi, My name is Pratik")
history.add_ai_message("Hello Pratik")

#ConversationBufferMemory
memory = ConversationBufferMemory()
memory.save_context({"input":"Hello"},{"output":"Hi"})

