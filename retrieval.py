
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_API_URL")

if not api_key or not base_url:
    raise ValueError(
        "Check OPENROUTER_API_KEY and OPENROUTER_API_URL in your .env file"
    )

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url

# Connect to the existing ChromaDB collection
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="health_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Accept a user question
question = input("Enter your health-related question: ")

# Retrieve the 3 most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3,
)

# Display retrieved chunks and source metadata
print("\nTop 3 relevant document chunks:\n")

if not results:
    print("No relevant chunks found.")
else:
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Content:")
        print(doc.page_content)

        print("\nSource metadata:")
        print(doc.metadata)

        print("\n" + "=" * 60 + "\n")
