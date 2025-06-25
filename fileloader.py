from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')


loader = TextLoader('sample.txt')
my_context = loader.load()

def load_model():
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        provider='featherless-ai',
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_model()

human_template = "{question}\n{doc}"

chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(human_template)
])

formatted_chat_prompt = chat_prompt.format_messages(
    question = "Date of birth of Modi?" ,
    doc = my_context
)

response = chat_model.invoke(formatted_chat_prompt)
print(response.content)



