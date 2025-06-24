import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("Project", "gemma-demo")

# Page settings
st.set_page_config(
    page_title="💎 Ask Gemma",
    page_icon="🧠",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #e0f7fa 0%, #f1f8e9 100%);
    }
    .main-container {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-top: 40px;
    }
    .chat-bubble {
        background: #d0f8ce;
        padding: 15px;
        border-radius: 15px;
        margin-top: 15px;
        font-size: 16px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    .typing {
        font-style: italic;
        opacity: 0.6;
        animation: blink 1s linear infinite;
    }
    @keyframes blink {
        50% { opacity: 0.1; }
    }
    .input-style input {
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 16px;
        border: 1px solid #b2dfdb !important;
    }
    .stButton>button {
        background-color: #26a69a;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("## 🌈 Ask Anything to Gemma")
st.markdown("Powered by **LangChain**, **Ollama**, and **LangSmith**")
st.markdown(f"🎯 Project: `{os.getenv('Project', 'gemma-demo')}`")

# --- Prompt Setup ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# --- User Input ---
st.markdown("### 💬 Ask your question below:")
input_text = st.text_input("Your Question", placeholder="e.g., Explain LangChain in simple terms", key="user_input", label_visibility="collapsed")

# --- Response Logic ---
if input_text:
    with st.spinner("🤔 Thinking..."):
        response = chain.invoke({"question": input_text})
        time.sleep(0.3)

    # Typing animation
    st.markdown('<div class="chat-bubble">', unsafe_allow_html=True)
    typing_placeholder = st.empty()
    full_response = ""

    for char in response:
        full_response += char
        typing_placeholder.markdown(full_response + "▌")
        time.sleep(0.01)

    typing_placeholder.markdown(full_response)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
