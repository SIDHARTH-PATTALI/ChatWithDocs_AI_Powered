import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("📄 Ask Your Documents using AI")

uploaded_files = st.file_uploader("📂 Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        documents = []


        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        vectors = FAISS.from_documents(split_docs, embeddings)

        # Save into session for retrieval
        st.session_state.vectors = vectors

if "vectors" in st.session_state:
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Prompt input
    user_prompt = st.text_input("💬 Ask a question from the uploaded documents")

    if user_prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed_time = round(time.process_time() - start, 2)

        st.write(f"🧠 **Answer:** {response['answer']}")
        st.write(f"⏱️ _Response Time: {elapsed_time} seconds_")

        with st.expander("📑 Similar Document Chunks"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("-" * 40)
else:
    st.info("Please upload at least one PDF document to begin.")

