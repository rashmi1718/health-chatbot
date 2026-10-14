
import os
import logging
import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Health Knowledge Chatbot",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Health Knowledge RAG Chatbot")
st.caption(
    "Ask questions about hydration, healthy diets, sleep, "
    "and infection prevention."
)

# Validate configuration
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_API_URL")

if not api_key or not base_url:
    st.error(
        "API configuration is missing. Check OPENROUTER_API_KEY "
        "and OPENROUTER_API_URL in your .env file."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url


# Initialize the RAG components once per Streamlit server process
@st.cache_resource
def initialize_rag():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name="health_knowledge",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    llm = ChatOpenAI(
        model=os.getenv(
            "OPENROUTER_CHAT_MODEL",
            "openai/gpt-4o-mini",
        ),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a health knowledge assistant providing general educational
information, not a medical diagnosis.

Answer the question using ONLY the retrieved context.

Rules:
- Do not invent facts or use outside knowledge.
- If the context is insufficient or irrelevant, clearly say:
  "The available documents do not contain enough information
  to answer this question."
- If only part of the question can be answered, explain what is missing.
- Keep the answer clear and concise.
- Remind the user that this information is educational and does not
  replace advice from a qualified healthcare professional.

Retrieved context:
{context}

Question:
{question}

Answer:
""")

    return vector_store, llm, prompt


# Retrieve documents and generate an answer
def get_answer(question, vector_store, llm, prompt):
    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return (
            "The available documents do not contain enough information "
            "to answer this question."
        ), []

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

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return response.content, sources


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages and their sources
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Source documents"):
                for source in message["sources"]:
                    st.write(f"- {source}")


# Accept a new question
question = st.chat_input("Ask a health-related question...")

if question:
    question = question.strip()

    if not question:
        st.warning("Please enter a question before submitting.")
    else:
        # Display and store the user's question
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    vector_store, llm, prompt = initialize_rag()

                    answer, sources = get_answer(
                        question,
                        vector_store,
                        llm,
                        prompt,
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Source documents"):
                            for source in sources:
                                st.write(f"- {source}")

                    # Store answer and sources for later reruns
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception:
                    logger.exception("Failed to process the question")
                    st.error(
                        "Sorry, I couldn't process your question. "
                        "Please check your API configuration, model "
                        "availability, and ChromaDB setup, then try again."
                    )

# Sidebar controls
with st.sidebar:
    st.subheader("Chat controls")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Answers are generated from the retrieved documents. "
        "This chatbot is for educational purposes only."
    )
