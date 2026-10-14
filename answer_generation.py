
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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

# 1. Connect to ChromaDB
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="health_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# 2. Configure the LLM
# Replace the model with one supported by your OpenRouter account.
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
    temperature=0,
)

# 3. Define a prompt that keeps answers grounded in context
prompt = ChatPromptTemplate.from_template("""
You are a health knowledge assistant providing general educational
information, not a medical diagnosis.

Answer the user's question using ONLY the retrieved context below.

Rules:
- Do not invent facts or use outside knowledge to fill gaps.
- If the context does not contain enough information, clearly say:
  "The available documents do not contain enough information to answer
  this question."
- If only part of the question can be answered, answer that part and
  clearly identify what information is missing.
- Include the source filenames used for the answer.
- If the context is irrelevant, explain that the available documents
  do not answer the question.
- Keep the answer clear and concise.
- Remind users that this information is educational, not a substitute
  for advice from a qualified healthcare professional.

Retrieved context:
{context}

User question:
{question}

Answer:
""")

# 4. Accept the user's question
question = input("Enter your health-related question: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# 5. Retrieve the top 3 relevant chunks
retrieved_docs = vector_store.similarity_search(
    question,
    k=3,
)

if not retrieved_docs:
    print("\nThe available documents do not contain enough information "
          "to answer this question.")
else:
    # 6. Prepare context and source filenames
    context_parts = []
    sources = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")

        context_parts.append(
            f"Source: {source}\nContent: {doc.page_content}"
        )

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # 7. Pass question and context to the LLM
    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    # 8. Display the answer and sources
    print("\n========== ANSWER ==========\n")
    print(response.content)

    print("\n========== SOURCE DOCUMENTS ==========\n")
    for source in sources:
        print(f"- {source}")
