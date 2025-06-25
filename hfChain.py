import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Load model
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

# Prompt 1: Code generation
gen_system = "You are helpful Python Software developer"
gen_human = "{question}"
prompt1 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(gen_system),
    HumanMessagePromptTemplate.from_template(gen_human)
])

# Prompt 2: Code optimization
opt_system = "You are Expert in Software developer"
opt_human = "optimize {code} in terms of time complexity and space complexity"
prompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(opt_system),
    HumanMessagePromptTemplate.from_template(opt_human)
])

# Chain setup
code_generation_chain = prompt1 | model | StrOutputParser()
optimize_code_chain = prompt2 | model | StrOutputParser()

# Streamlit UI
st.title("💡 Python Code Generator + Optimizer")

question = st.text_area("🧠 Enter your task description", "Longest Substring Without Repeating Characters in python?")

if st.button("Generate and Optimize"):
    with st.spinner("Generating code..."):
        generated_code = code_generation_chain.invoke({'question': question}).strip()

    with st.spinner("Optimizing code..."):
        optimized_code = optimize_code_chain.invoke({'code': generated_code}).strip()

    st.subheader("🧾 Generated Code:")
    st.code(generated_code, language='python')

    st.subheader("🛠️ Optimized Code:")
    if optimized_code == generated_code:
        st.info("✅ The generated code is already optimal.")
    else:
        st.code(optimized_code, language='python')
