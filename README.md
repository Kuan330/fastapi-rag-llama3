# Local-Hosted RAG Knowledge Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Llama3](https://img.shields.io/badge/Model-Llama3--8B-purple)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## Overview

This project is a production-ready **Retrieval-Augmented Generation (RAG)** API designed to ingest technical documents (PDFs) and allow users to query them using Large Language Models (LLMs) running entirely on local hardware.

Unlike cloud-based solutions, this engine prioritizes **data privacy** and **inference latency** by leveraging local NVIDIA GPU acceleration (RTX 4060 Ti) and quantized models (Llama 3 8B) via Ollama. It wraps the core inference logic in a scalable **FastAPI** backend, making it ready for integration with frontend applications or microservices architectures.

## Key Features

* **Zero-Data-Leakage:** All processing, from embedding generation to inference, happens locally. No data is sent to external APIs (OpenAI/Anthropic).
* **Vector Search:** Utilizes **ChromaDB** and **FAISS** for high-performance similarity search on high-dimensional text embeddings.
* **Scalable API:** Built with **FastAPI** for asynchronous request handling and automatic documentation (Swagger UI).
* **Containerized:** Includes Docker support for consistent deployment across environments.
* **Hardware Accelerated:** Optimized for consumer-grade GPUs (NVIDIA RTX Series).

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | Llama 3 (via Ollama) | State-of-the-art 8B open-source model. |
| **Orchestration** | LangChain | Manages the retrieval chains and prompt engineering. |
| **API Framework** | FastAPI | High-performance web framework for building APIs with Python. |
| **Vector Store** | ChromaDB | Database for storing and retrieving vector embeddings. |
| **Containerization** | Docker | Ensures environment consistency. |

## Installation & Setup

### Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) installed and running.
* NVIDIA GPU (Recommended for performance).

### 1. Clone the Repository
```bash
git clone [https://github.com/Kuan330/fastapi-rag-llama3.git](https://github.com/Kuan330/fastapi-rag-llama3.git)
cd fastapi-rag-llama3
```

### 2. Set up the Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


### 3. Pull the Model
Ensure Ollama is running, then pull the Llama 3 model:
```bash
ollama pull llama3
```


### 4. Run the API
```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000.

### Usage
Navigate to the Swagger UI at http://127.0.0.1:8000/docs.
Use the /upload endpoint to upload a PDF file (e.g., a research paper or manual).
Use the /query endpoint to ask questions about the uploaded document.

### Docker Deployment
To build and run the application as a container:
```bash
docker build -t rag-engine .
docker run -p 8000:8000 rag-engine
```

### Future Improvements
Hybrid Search: Implementing BM25 keyword search alongside vector search for better retrieval accuracy.
Multi-Modal Support: Adding support for image-based PDFs using LlaVa.
Frontend: Building a simple Streamlit or React UI for easier interaction.

Built by Lee Kuan Loong
