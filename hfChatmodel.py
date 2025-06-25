# app.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

# Initialize model
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_model()

# Streamlit UI
st.title("🧠 LLM Chatbot with Hugging Face")

user_input = st.text_input("Ask something:", placeholder="Who are you?")

if user_input:
    with st.spinner("Thinking..."):
        response = chat_model.invoke(user_input)
        st.markdown(f"**Response:** {response.content}")
