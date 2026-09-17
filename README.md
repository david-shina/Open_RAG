# Django RAG API

A high-performance Django REST Framework API designed for **Retrieval-Augmented Generation (RAG)** tasks. This system allows users to upload documents (PDF/Text), automatically generates vector embeddings, and performs intelligent question-answering using advanced query translation techniques (Multi-query, Step-back) and Large Language Models (LLMs).

## 🚀 Key Features

* **Document Ingestion:** Upload and process PDF, DOCX, and TXT files.
* **Smart Chunking:** Automatically splits documents into semantic chunks for better retrieval.
* **Vector Embeddings:** Integration with **Ollama** (using `nomic-embed-text`) for local, high-quality embeddings.
* **Advanced Query Translation:**
    * **Multi-Query:** Generates multiple variations of a user's question to overcome keyword mismatch.
    * **Step-Back Prompting:** Generates abstract, higher-level questions to retrieve broader context.
    * **Hybrid Mode:** Combines strategies for maximum recall.
* **LLM Integration:** Powered by **Groq** (openai/gpt-oss-120b) for ultra-fast answer generation.
* **Interactive Documentation:** Fully documented API with **Swagger UI** (via `drf-spectacular`).

## 🛠️ Tech Stack

* **Backend:** Django, Django REST Framework (DRF)
* **AI/Orchestration:** LangChain
* **LLM Provider:** Groq API (openai/gpt-oss-120b)
* **Embeddings:** Ollama (Nomic Embed Text)
* **Vector Store:** In-memory / FAISS (configurable)
* **Documentation:** drf-spectacular (OpenAPI 3.0)

---

## ⚙️ Prerequisites

1.  **Python 3.10+**
2.  **Ollama** installed locally (for embeddings).
    * Run: `ollama pull nomic-embed-text`
3.  **Groq API Key** (for the LLM). Get one at [console.groq.com](https://console.groq.com).

---

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Luvakin/Open_RAG.git](https://github.com/Luvakin/Open_RAG.git)
    cd OPEN_RAG_FINAL
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory:
    ```ini
    DEBUG=True
    SECRET_KEY=your-super-secret-key-change-me
    
    # AI Configuration
    GROQ_API_KEY=gsk_your_groq_api_key_here
    LANGCHAIN_API_KEY=your_langchain_key
    TAVILY_API_KEY=tavily_api_key(web search skill in the works)
    
    ```

5.  **Run Migrations**
    ```bash
    python manage.py migrate
    ```

6.  **Start the Server**
    ```bash
    python manage.py runserver
    ```

---

## 📖 API Documentation

Once the server is running, you can access the interactive API documentation to test endpoints directly from your browser.

* **Swagger UI:** [http://localhost:8000/api/docs/](http://localhost:8000/api/docs/)
* **ReDoc:** [http://localhost:8000/api/redoc/](http://localhost:8000/api/redoc/)

---

## ⚡ Usage Examples

### 1. Upload a Document
**Endpoint:** `POST /api/upload/`
* **Content-Type:** `multipart/form-data`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | File | The PDF or Text document to analyze. |
| `user_id` | String | Unique identifier for the user session. |

**Response:**
```json
{
  "success": true,
  "message": "Document uploaded and processed successfully",
  "document_id": "research_paper.pdf",
  "chunks_created": 42
}
```

### 2. Query Document
**Endpoint:** `POST /api/query/`
* **Content-Type:** `multipart/form-data`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `question` | String | document_question |
| `user_id` | String | Unique identifier for the user session. |
| `document_id` | String | Name of document from the /upload/ response |


**Response:**
```json
{
    "success": true,
    "original_query":"question",
    "answer": "final_answer",
    "generated_queries": "generated_queries",
    "source_chunks": "len(retrieved_docs_list)"
}