
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_API_URL")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

if not base_url:
    raise ValueError("OPENROUTER_API_URL is missing from .env")

# Configure OpenRouter
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url

# Load TXT documents from doc_ref
txt_loader = DirectoryLoader(
    "doc_ref",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)

documents = txt_loader.load()

if not documents:
    raise ValueError("No TXT documents found in the doc_ref folder")

print(f"Loaded {len(documents)} documents")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Store chunks in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="health_knowledge",
)

print("Documents successfully stored in ChromaDB!")
