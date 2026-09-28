import os
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" not in os.environ:
    st.error(
        "GOOGLE_API_KEY not found. Add it to .streamlit/secrets.toml as:\n\n"
        'GOOGLE_API_KEY = "your-key-here"\n\n'
        "or set it as an environment variable before launching."
    )
    st.stop()

GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "equity_analyst_chroma")

st.set_page_config(page_title="AI Equity Analyst",
                   page_icon="📈", layout="wide")
st.title("📈 AI Equity Analyst")
st.caption(
    "Upload one or more financial reports (10-K, 10-Q, earnings releases) "
    "and ask questions — including comparisons across documents."
)


# -----------------------------------------------------------------------
# CACHED RESOURCES
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


# -----------------------------------------------------------------------
# DOCUMENT PROCESSING
# -----------------------------------------------------------------------
def process_pdf(uploaded_file):
    """Load and chunk a single uploaded PDF. Returns a list of chunks
    (each tagged with its source filename) or None on failure."""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        if not docs or all(not d.page_content.strip() for d in docs):
            st.error(
                f"No extractable text found in '{uploaded_file.name}'. "
                "It may be a scanned image-only PDF — try an OCR'd version."
            )
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits = text_splitter.split_documents(docs)

        # Tag every chunk with which file it came from and its page number.
        # This is what lets citations say "[Q1_Report.pdf, p.3]" instead of
        # just "[p.3]" once multiple documents are mixed in the same store —
        # without this, retrieved chunks from different files are indistinguishable.
        for chunk in splits:
            chunk.metadata["source_file"] = uploaded_file.name
            chunk.metadata["page_display"] = chunk.metadata.get("page", 0) + 1

        return splits

    except Exception as e:
        st.error(f"Failed to process '{uploaded_file.name}': {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def build_rag_chain(retriever):
    """History-aware retrieval chain with analyst-style, multi-document-aware prompting."""
    llm = get_llm()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might "
         "reference context in the chat history, formulate a standalone "
         "question which can be understood without the chat history. Do "
         "NOT answer the question, just reformulate it if needed and "
         "otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Analyst-style system prompt: explicitly asks for comparison/synthesis
    # across documents when more than one is present, not just fact lookup.
    # Retrieved chunks are tagged with source_file/page_display in metadata,
    # so the model is told to use that to attribute and compare correctly.
    system_prompt = (
        "You are an expert equity/financial analyst. Use the retrieved "
        "context — which may come from ONE OR MORE financial reports — to "
        "answer the user's question.\n\n"
        "Each piece of context is preceded by its source filename and page "
        "number in the format [source: FILENAME, p.PAGE]. Use these to "
        "attribute figures correctly and never mix numbers from different "
        "documents without saying so.\n\n"
        "Rules:\n"
        "- Cite sources inline using the format [FILENAME, p.X] for every "
        "figure or claim you state.\n"
        "- When asked about specific numbers (revenue, EPS, margins, etc.), "
        "quote the exact value and unit as reported — do not round or "
        "estimate.\n"
        "- If the context includes MULTIPLE source documents (e.g. two "
        "periods, or two companies), proactively compare and contrast them "
        "even if not explicitly asked — call out what changed, improved, "
        "worsened, or diverged, and by how much where the numbers allow it.\n"
        "- If asked for an opinion or assessment (e.g. 'is this a strong "
        "quarter'), reason from the retrieved figures and trends rather "
        "than giving a generic answer — state what specifically supports "
        "your view.\n"
        "- If the retrieved context does not contain the answer, say so "
        "clearly instead of guessing.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # document_prompt controls how EACH retrieved chunk is formatted before
    # being joined into {context} — this is what actually injects the
    # [source: FILENAME, p.PAGE] tag ahead of every chunk's text, so the
    # LLM can see provenance per-chunk, not just as one undifferentiated blob.
    document_prompt = ChatPromptTemplate.from_template(
        "[source: {source_file}, p.{page_display}]\n{page_content}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt, document_prompt=document_prompt
    )

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# -----------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_files" not in st.session_state:
    # names of files already embedded this session
    st.session_state.processed_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# -----------------------------------------------------------------------
# UI — UPLOAD (multiple files)
# -----------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload financial reports (PDF) — one or more",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_files = [
        f for f in uploaded_files if f.name not in st.session_state.processed_files]

    if new_files:
        with st.spinner(f"Reading and embedding {len(new_files)} document(s)..."):
            all_new_splits = []
            for f in new_files:
                splits = process_pdf(f)
                if splits:
                    all_new_splits.extend(splits)
                    st.session_state.processed_files.append(f.name)

            if all_new_splits:
                if st.session_state.vectorstore is None:
                    # First document(s) in this session — fresh Chroma
                    # collection. Wipe any leftover dir from a prior run.
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=all_new_splits,
                        embedding=get_embeddings(),
                        persist_directory=CHROMA_DIR,
                    )
                else:
                    # ADD to the existing store instead of recreating it —
                    # this is the core change that enables multi-document
                    # support: prior files' vectors stay searchable.
                    st.session_state.vectorstore.add_documents(all_new_splits)

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 25},
                )
                st.session_state.rag_chain = build_rag_chain(retriever)
                st.success(
                    f"{len(st.session_state.processed_files)} document(s) ready to query.")

    if st.session_state.processed_files:
        st.markdown("**Loaded documents:** " +
                    ", ".join(f"`{n}`" for n in st.session_state.processed_files))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🧹 Clear all documents"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # -------------------------------------------------------------------
    # UI — CHAT
    # -------------------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder_text = (
        "Ask a question — try a comparison, e.g. 'How did revenue change "
        "between these reports?'" if len(st.session_state.processed_files) > 1
        else "Ask a question about the uploaded report..."
    )
    user_query = st.chat_input(placeholder_text)

    if user_query and st.session_state.rag_chain is not None:
        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for chunk in st.session_state.rag_chain.stream({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history,
                }):
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error generating answer: {e}"
                response_placeholder.error(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=full_response))

        MAX_TURNS = 10
        if len(st.session_state.chat_history) > MAX_TURNS * 2:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_TURNS * 2:]

else:
    st.info("👆 Upload one or more PDF financial reports to get started. Uploading multiple lets you ask comparison questions across them.")
