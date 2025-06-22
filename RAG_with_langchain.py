
# Projekt: Interaktive PDF-Frage-Antwort-App mit LangChain und Streamlit
# ----------------------------------------
# Benötigte Pakete für PDF-QA mit LangChain
# ----------------------------------------

# Streamlit für das Web-Interface
# pip install streamlit

# LangChain als Haupt-Framework
# pip install langchain

# OpenAI API (für GPT-Modelle)
# pip install openai

# PDF-Verarbeitung (für PyPDFLoader)
# pip install pypdf

# Vektor-Datenbank (lokal) via Chroma
# pip install chromadb

# Falls du Umgebungsvariablen per Datei laden willst:
# pip install python-dotenv

# Optional für stabilere Textverarbeitung
# pip install tiktoken

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os

# Streamlit Interface
st.set_page_config(page_title="PDF Q&A with LangChain", layout="wide")
st.title("📄 PDF Question Answering with LangChain")

# OpenAI API Key (alternativ über Umgebungsvariable)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Session-State für Verlauf
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if uploaded_file and openai_api_key:
    # Temporäre Datei speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # 1. PDF laden
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. In Chunks teilen
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # 3. Embeddings + Vektor-Datenbank
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # 4. Retrieval QA Chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 5. Frage stellen
    question = st.text_input("Stelle eine Frage zum Dokument:")
    if question:
        with st.spinner("Suche nach Antwort..."):
            answer = qa_chain.run(question)
        st.session_state.qa_history.append({"Frage": question, "Antwort": answer})
        st.markdown("### Antwort:")
        st.write(answer)

    # Verlauf anzeigen und bearbeiten
    if st.session_state.qa_history:
        st.markdown("---")
        st.subheader("📜 Verlauf")
        for i, entry in enumerate(st.session_state.qa_history):
            with st.expander(f"Frage {i+1}: {entry['Frage']}"):
                st.write("Antwort:", entry["Antwort"])
                new_q = st.text_input(f"Frage bearbeiten {i+1}", value=entry["Frage"], key=f"edit_q_{i}")
                if new_q != entry["Frage"]:
                    new_answer = qa_chain.run(new_q)
                    st.session_state.qa_history[i] = {"Frage": new_q, "Antwort": new_answer}
                    st.experimental_rerun()
else:
    st.info("Bitte lade ein PDF hoch und gib deinen OpenAI API Key ein.")
