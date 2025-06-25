import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Load the model
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_model()

# Supported languages
languages = [
    "english", "hindi", "french", "german", "spanish", "chinese", "japanese", 
    "korean", "arabic", "russian", "marathi", "tamil", "telugu", "gujarati"
]

# Streamlit UI
st.title("🌍 LLM Translator")

st.subheader("Translate a phrase between languages")

topic = st.text_input("Enter text to translate", "how are you?")
lang1 = st.selectbox("Translate from", options=languages, index=0)
lang2 = st.selectbox("Translate to", options=languages, index=1)

if st.button("Translate"):
    if lang1 == lang2:
        st.warning("Source and target languages must be different.")
    else:
        prompt = PromptTemplate(
            input_variables=["topic", "lang1", "lang2"],
            template="{topic} Translate this from {lang1} to {lang2}"
        )
        formatted_prompt = prompt.format(topic=topic, lang1=lang1, lang2=lang2)
        with st.spinner("Translating..."):
            response = chat_model.invoke(formatted_prompt)
            st.markdown(f"**Translation:**\n\n{response.content}")
