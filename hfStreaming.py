import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

# Load .env variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

# Load model
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        streaming=True
    )
    return ChatHuggingFace(llm=llm)

chat = load_model()

# UI
st.title("🧠 Streaming LLM Storyteller")
topic = st.text_input("Enter a topic for the story:", "India")

if st.button("Generate Story"):
    st.write("### Story Output:")
    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("Tell me a long story about {topic}")
    ])
    formatted_chat_prompt = chat_prompt.format_messages(topic=topic)

    # Output area for streaming
    output_placeholder = st.empty()
    story = ""
    for chunk in chat.stream(formatted_chat_prompt):
        story += chunk.content
        output_placeholder.markdown(story)
