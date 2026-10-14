
import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Health Knowledge RAG Chatbot",
    description="Answers health education questions using retrieved documents.",
    version="1.0.0",
)

# Request schema
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The health-related question to ask",
    )


# Validate configuration
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_API_URL")

if not api_key or not base_url:
    raise RuntimeError(
        "OPENROUTER_API_KEY or OPENROUTER_API_URL is missing from .env"
    )

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Connect to existing ChromaDB collection
vector_store = Chroma(
    collection_name="health_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Initialize chat model
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
)

# Prompt for grounded answers
prompt = ChatPromptTemplate.from_template("""
You are a health knowledge assistant providing general educational
information, not a medical diagnosis.

Answer using ONLY the retrieved context below.

Rules:
- Do not invent facts or use outside knowledge.
- If the context is insufficient or irrelevant, clearly say:
  "The available documents do not contain enough information
  to answer this question."
- If only part of the question can be answered, explain what is missing.
- Keep the answer clear and concise.
- Remind the user that this is educational information, not a
  substitute for advice from a qualified healthcare professional.

Retrieved context:
{context}

Question:
{question}

Answer:
""")


@app.get("/")
def home():
    return {"message": "Health Knowledge RAG Chatbot API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    # Handle whitespace-only questions
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        # Retrieve the 3 most relevant chunks
        docs = vector_store.similarity_search(question, k=3)

        if not docs:
            return {
                "question": question,
                "answer": (
                    "The available documents do not contain enough "
                    "information to answer this question."
                ),
                "sources": [],
            }

        # Build context and collect unique source filenames
        context_parts = []
        sources = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")

            context_parts.append(
                f"Source: {source}\nContent: {doc.page_content}"
            )

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        # Generate grounded answer
        messages = prompt.format_messages(
            context=context,
            question=question,
        )

        response = llm.invoke(messages)

        return {
            "question": question,
            "answer": response.content,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Error processing chat request")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the answer.",
        )
