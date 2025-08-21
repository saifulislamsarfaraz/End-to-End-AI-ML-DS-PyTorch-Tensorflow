# 🤖 LLM Query Prototype

This project is an **LLM-powered Query System** that allows users to interact with an advanced AI assistant. It uses **FastAPI** as the backend and **Streamlit** as the frontend to provide a seamless interface for querying and retrieving context-aware responses.

---

## 🚀 Features

- **Interactive Query Interface**:
  - Users can input questions and optional context to get detailed responses.
  - Built with **Streamlit** for a user-friendly experience.

- **FastAPI Backend**:
  - Handles user queries and integrates retrieval-augmented generation (RAG) with a lightweight LLM.

- **Retrieval-Augmented Generation (RAG)**:
  - Retrieves relevant context from a vector database (FAISS) to enhance the accuracy of responses.

- **Lightweight LLM**:
  - Uses a quantized version of the **TinyLlama** model for fast and efficient inference.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **Generative Model**: TinyLlama (CTranslate2)

---

## 🖥️ Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Required Python libraries (see `requirements.txt`)

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd Hackathon

uvicorn app:app --reload --host 127.0.0.1 --port 8000

streamlit run [streamlit_app.py](http://_vscodecontentref_/3)

curl -X GET "http://127.0.0.1:8000/query?question=What+are+the+top+reviews+for+dresses?"