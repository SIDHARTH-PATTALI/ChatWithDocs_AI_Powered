# 📄 ChatWithDocs

**ChatWithDocs** is a Streamlit-based AI assistant that allows you to upload one or more PDF documents and ask questions about their content using powerful Large Language Models (LLMs). It uses Groq's LLaMA model for answering and Hugging Face embeddings for document understanding.

---

## 🚀 Features

- 🧠 Chat with one or more PDF documents
- 👢 Multiple file upload support
- ⚡ Fast document search using FAISS vector store
- 🗞️ Accurate chunk-based retrieval and answer generation
- 🔐 API key-based authentication with Groq
- 🖼️ Simple and clean Streamlit UI

---

## 🛠️ Tech Stack

| Layer              | Tools Used                                                    |
|-------------------|----------------------------------------------------------------|
| 🧠 LLM             | Groq's `llama-3.3-70b-versatile` via LangChain               |
| 📄 Document Loader | `PyPDFLoader` from LangChain Community                      |
| 📊 Embeddings      | `HuggingFaceEmbeddings` with MiniLM model                  |
| 🔍 Vector DB       | FAISS                                                      |
| 💬 QA Pipeline     | LangChain Retrieval Chain                                  |
| 🌐 Frontend        | Streamlit                                                  |

---

## 🔍 How It Works

1. Upload one or more PDF files.
2. Text is extracted using `PyPDFLoader`.
3. Text is split into chunks.
4. Embeddings are generated using Hugging Face MiniLM model.
5. Chunks are stored in FAISS for fast similarity search.
6. A retrieval-based QA chain fetches relevant chunks.
7. Groq's LLaMA model generates responses.

---

## 📅 Example Usage

Run the app locally:

```bash
streamlit run main.py
```

Upload any PDF (e.g., resume, report, article) and ask:

- "What is the main idea of this paper?"
- "Summarize this section."
- "What does the author say about XYZ?"

---

## ✅ Dependencies

```txt
streamlit
langchain
dotenv
faiss-cpu
langchain_groq
langchain_community
sentence-transformers
```

You can install them using:

```bash
pip install streamlit langchain python-dotenv faiss-cpu langchain_groq langchain_community sentence-transformers
```

---



