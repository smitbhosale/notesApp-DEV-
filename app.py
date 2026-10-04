import streamlit as st
from pypdf import PdfReader
import ollama
import chromadb

# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="FriendStudy",
    page_icon="📚",
    layout="centered"
)

# -----------------------------
# CHROMA
# -----------------------------

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="friendstudy"
)

# -----------------------------
# FUNCTIONS
# -----------------------------

def extract_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=800):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


def create_embeddings(chunks):

    for i, chunk in enumerate(chunks):

        embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )["embedding"]

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
        )


def retrieve_context(question):

    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )["embedding"]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=4
    )

    documents = results["documents"][0]

    return "\n\n".join(documents)


def ask_ai(question, context):

    prompt = f"""
You are FriendStudy, a personal AI tutor.

Answer the student's question using the study material below.

Rules:

- Use the provided context.
- Explain things simply.
- Give examples when useful.
- Do not invent facts.
- If the answer isn't present in the context, say so.
- Structure the answer clearly.

STUDY MATERIAL:

{context}

QUESTION:

{question}
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# -----------------------------
# UI
# -----------------------------

st.title("📚 FriendStudy")

st.caption(
    "Your private AI study companion — powered by local AI."
)

uploaded_file = st.file_uploader(
    "📄 Upload your study notes",
    type=["pdf"]
)

# -----------------------------
# PROCESS PDF
# -----------------------------

if uploaded_file:

    if st.button("Process Notes", type="primary"):

        with st.spinner("Reading your notes..."):

            text = extract_pdf(uploaded_file)

            chunks = chunk_text(text)

            create_embeddings(chunks)

        st.success(
            f"✅ Processed {len(chunks)} sections from your notes!"
        )

# -----------------------------
# QUESTION
# -----------------------------

question = st.text_input(
    "Ask something about your notes",
    placeholder="Explain DBSCAN in simple terms..."
)

if question:

    with st.spinner("Thinking..."):

        context = retrieve_context(question)

        answer = ask_ai(
            question,
            context
        )

    st.markdown("### 🤖 FriendStudy")

    st.write(answer)