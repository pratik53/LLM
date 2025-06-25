import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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
    "english", "hindi", "french", "german", "spanish", "chinese", 
    "japanese", "korean", "arabic", "russian", "marathi", "tamil"
]

# Streamlit UI
st.title("🌐 LLM Translator using ChatPromptTemplate")
st.markdown("This app uses a system + human message structure with `ChatPromptTemplate`.")

# User input
text = st.text_input("Enter text to translate", "Who are you?")
input_language = st.selectbox("Translate from", languages, index=0)
output_language = st.selectbox("Translate to", languages, index=1)

# Button
if st.button("Translate"):
    if input_language == output_language:
        st.warning("Source and target languages must be different.")
    else:
        # You can use this
    
        #chat_prompt = ChatPromptTemplate.from_messages([
            #("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
            #("human", "{text}")
        #])


        # You can also use this
        sys_template = "You are a helpful assistant that translates {input_language} to {output_language}."
        human_template = "{text}"
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sys_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        formatted_chat_prompt = chat_prompt.format_messages(
            input_language=input_language,
            output_language=output_language,
            text=text
        )

        # Get LLM response
        with st.spinner("Translating..."):
            response = chat_model.invoke(formatted_chat_prompt)
            st.markdown(f"**Translation:**\n\n{response.content}")
