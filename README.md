# Local-Hosted RAG Knowledge Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Llama3](https://img.shields.io/badge/Model-Llama3--8B-purple)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## Overview

This project is a production-ready **Retrieval-Augmented Generation (RAG)** API designed to ingest technical documents (PDFs) and allow users to query them using Large Language Models (LLMs) running entirely on local hardware.

Unlike simple wrappers, this engine prioritizes **inference latency** and **retrieval accuracy** by leveraging **OS-Aware Hardware Acceleration**. It automatically detects the host system (NVIDIA GPU vs. Apple Silicon) to optimize context windows and quantization settings dynamically.

## Key Features

* **Privacy First:** Runs 100% offline. No data is sent to OpenAI/Anthropic.
* **Hardware Aware:** Automatically detects host hardware to optimize performance:
    * **Windows (NVIDIA 4060 Ti):** Unlocks 8k context window & ultra-high retrieval (25 chunks).
    * **Mac (M1/M2/M3):** Optimizes for Unified Memory (4k context) to prevent swap thrashing.
* **Advanced Retrieval:** Utilizes **Maximal Marginal Relevance (MMR)** instead of basic similarity search to reduce redundancy and hallucination.
* **Self-Healing:** Automatically detects and downloads missing model weights (`nomic-embed-text`) on startup.
* **Dockerized:** Deployment-ready with a single container build that connects seamlessly to the host's Ollama instance.

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | Llama 3 (via Ollama) | State-of-the-art 8B open-source model. |
| **Embeddings** | Nomic-Embed-Text | Optimized high-performance embedding model. |
| **Orchestration** | LangChain v0.3 | Manages retrieval chains and modern LCEL architecture. |
| **API Framework** | FastAPI | Asynchronous request handling with automatic Swagger UI docs. |
| **Vector Store** | ChromaDB | Persistent local vector storage. |
| **Containerization** | Docker | Production-grade container deployment. |

## Quick Start (Docker)

The easiest way to run this engine is via Docker, which handles all dependencies (LangChain, ChromaDB, Drivers).

### 1. Prerequisites
* [Docker Desktop](https://www.docker.com/) installed.
* [Ollama](https://ollama.com/) installed and running.

### 2. Pull Required Models
Ensure your local Ollama instance has the brains it needs:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Build & Run
```bash
# Build the container
docker build -t rag-engine .

# Run the container
# Note: We use 'host.docker.internal' so Docker can talk to your local Ollama
docker run -p 8000:8000 -e OLLAMA_BASE_URL="[http://host.docker.internal:11434](http://host.docker.internal:11434)" rag-engine
```

### 4. Use the API
Open your browser to the auto-generated Swagger UI: http://localhost:8000/docs

POST /upload: Upload a PDF file. The engine will clean, chunk, and index it.

POST /query: Ask a question. The engine will retrieve the top relevant chunks using MMR and stream the answer.

### Manual Installation (Dev Mode)
If you want to edit the code, you can run it without Docker:
```bash
# 1. Create Environment
conda create -n rag-env python=3.10
conda activate rag-env

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run Server
python -m uvicorn main:app --reload
```

### Future Improvements
Hybrid Search: Implement BM25 keyword search alongside vector search for better precision.

Multi-Modal: Add support for image-based PDFs using LlaVa.

Frontend: Build a simple Streamlit UI for non-technical users.

### Built by Lee Kuan Loong
