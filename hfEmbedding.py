from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # or any other supported model
)

# Embed a single text
vector = embeddings.embed_query("This is a test sentence.")
print(vector)

# Embed a list of documents
docs = ["Hello world", "LangChain is awesome"]
vectors = embeddings.embed_documents(docs)
print(vectors)
