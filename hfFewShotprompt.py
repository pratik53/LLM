import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Load LLM
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_model()

# Example Q&A
examples = [
    {
        'input': "Write a simple Python program to print an output.",
        'output': 'print("Hello, World!")'
    },
    {
        'input': "Write a Python code to check if a number is even or odd",
        'output': '''def is_even(num):\n    return num % 2 == 0'''
    }
]

# Prompt format for each example
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input:\n{input}\nOutput:\n{output}\n"
)

# Few-shot prompt template
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a helpful assistant that writes Python code based on the input description.\n\nHere are some examples:",
    suffix="Input:\n{user_input}\nOutput:",
    input_variables=["user_input"]
)

# --- Streamlit UI ---
st.title("💻 Python Code Generator (Few-Shot Prompt)")

user_input = st.text_area("Describe the Python task", "Write a Python function to calculate factorial.")

if st.button("Generate Code"):
    formatted_prompt = prompt_template.format(user_input=user_input)
    
    with st.spinner("Generating..."):
        response = chat_model.invoke(formatted_prompt)
        st.subheader("🔽 Output:")
        st.code(response.content, language="python")
