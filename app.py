import streamlit as st
import os

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from google import genai


# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# ==========================================
# 2. Load Gemini API Key
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:

    st.error(
        "Gemini API key not found. "
        "Please check your .env file."
    )

    st.stop()


client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ==========================================
# 3. Initialize Chat History
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 4. Initialize Vector Database
# ==========================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_characters" not in st.session_state:
    st.session_state.pdf_characters = 0

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0


# ==========================================
# 5. Main Header
# ==========================================

st.title("📚 PDF RAG Chatbot")

st.subheader("Chat with your PDF using AI")

st.info(
    "Upload a PDF and ask questions about its content."
)


# ==========================================
# 6. Sidebar
# ==========================================

with st.sidebar:

    st.header("📄 PDF Settings")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    st.divider()

    st.header("ℹ️ About")

    st.write(
        "This chatbot uses RAG to search your PDF "
        "and answer questions using AI."
    )

    st.divider()

    st.subheader("📊 PDF Information")

    if st.session_state.pdf_name:

        st.success("PDF Ready ✅")

        st.write(
            f"**File:** {st.session_state.pdf_name}"
        )

        st.write(
            f"**Characters:** "
            f"{st.session_state.pdf_characters}"
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.num_chunks}"
        )

    else:

        st.warning(
            "No PDF uploaded yet."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# 7. Process PDF
# ==========================================

if uploaded_file is not None:

    # Process only when a new PDF is uploaded
    if st.session_state.pdf_name != uploaded_file.name:

        # --------------------------------------
        # Read PDF
        # --------------------------------------

        with st.spinner(
            "📖 Reading your PDF..."
        ):

            reader = PdfReader(
                uploaded_file
            )

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text


        # Save PDF information

        st.session_state.pdf_name = (
            uploaded_file.name
        )

        st.session_state.pdf_characters = (
            len(text)
        )


        # --------------------------------------
        # Split PDF into Chunks
        # --------------------------------------

        with st.spinner(
            "✂️ Splitting PDF into chunks..."
        ):

            text_splitter = (
                RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
            )

            chunks = (
                text_splitter.split_text(text)
            )


        st.session_state.num_chunks = (
            len(chunks)
        )


        # --------------------------------------
        # Create Embeddings
        # --------------------------------------

        with st.spinner(
            "🧠 Creating embeddings..."
        ):

            embeddings = HuggingFaceEmbeddings(
                model_name=(
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                )
            )


        # --------------------------------------
        # Create FAISS Database
        # --------------------------------------

        with st.spinner(
            "🔎 Creating search database..."
        ):

            vectorstore = FAISS.from_texts(
                chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = (
                vectorstore
            )


        # Clear previous chat

        st.session_state.messages = []

        st.success(
            "✅ PDF is ready! "
            "You can now ask questions."
        )


# ==========================================
# 8. Display Previous Messages
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ==========================================
# 9. Chat Input
# ==========================================

if st.session_state.vectorstore is not None:

    question = st.chat_input(
        "💬 Ask something about your PDF..."
    )

    if question:

        # --------------------------------------
        # Show User Question
        # --------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.write(question)


        # --------------------------------------
        # Search PDF
        # --------------------------------------

        with st.spinner(
            "🔎 Searching your PDF..."
        ):

            results = (
                st.session_state.vectorstore
                .similarity_search(
                    question,
                    k=3
                )
            )


        # --------------------------------------
        # Create Context
        # --------------------------------------

        context = "\n\n".join(
            result.page_content
            for result in results
        )


        # --------------------------------------
        # Create Prompt
        # --------------------------------------

        prompt = f"""
You are a helpful PDF assistant.

Answer the user's question clearly and concisely
using ONLY the information from the uploaded PDF.

Do not mention "context", "retrieval", "chunks",
or how the answer was generated.

If the answer is not found in the PDF, say:

"I couldn't find this information in the uploaded document."

PDF Information:
{context}

User Question:
{question}

Answer:
"""


        # --------------------------------------
        # Ask Gemini
        # --------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                response = (
                    client.interactions.create(
                        model="gemini-3.5-flash-lite",
                        input=prompt
                    )
                )

                answer = response.output_text

            st.write(answer)


        # --------------------------------------
        # Save Assistant Response
        # --------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ==========================================
# 10. No PDF Message
# ==========================================

else:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "to start chatting."
    )