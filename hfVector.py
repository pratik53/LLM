from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#load
loader = TextLoader('sample.txt')
context = loader.load()

#split
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
context_doc = text_splitter.split_documents(context)

#Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # or any other supported model
)

#store
db = Chroma.from_documents(context_doc, embeddings, persist_directory="./chromedb")
db.persist()

#Read from chromadb
db_connection = Chroma(embedding_function=embeddings, persist_directory="./chromedb")

retriver = db.as_retriever()
query = "After Indira,"
#User for similar search
#similar_docs = db.similarity_search(query)
#you can also use this
similar_docs = retriver.get_relevant_documents(query)

#You can get directly from DB
retriver = db_connection.as_retriever()
similar_docs = retriver.get_relevant_documents(query)
print(similar_docs[0].page_content)