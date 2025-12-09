import os
import shutil
import platform
import sys
from tqdm import tqdm  # Progress bar library
import ollama  # need the direct client to check models

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")

def ingest_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")

    print(f" Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks from the document.")
    return splits

def create_vector_db(splits):
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    
    # FIX: Define the URL *first* so Docker knows where to look
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # AUTO-FIX: Check if the model exists (using the correct URL)
    required_model = "nomic-embed-text"
    try:
        # create a specific client pointing to the right address
        client = ollama.Client(host=OLLAMA_URL)
        client.show(required_model)
    except ollama.ResponseError:
        print(f"Model '{required_model}' not found. Downloading it now... (This may take a minute)")
        try:
            client = ollama.Client(host=OLLAMA_URL)
            client.pull(required_model)
            print(f"Download complete!")
        except Exception as e:
            print(f"FAILED to download model automatically: {e}")
            print("Please run 'ollama pull nomic-embed-text' manually on your host machine.")
    except Exception as e:
        print(f"Warning: Could not connect to Ollama at {OLLAMA_URL} to check model status.")
        print(f"Error: {e}")

    print(f"Creating vector embeddings for {len(splits)} chunks")
    
    # Now we use that same URL for LangChain
    embeddings = OllamaEmbeddings(model=required_model, base_url=OLLAMA_URL)
    
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    batch_size = 10
    for i in tqdm(range(0, len(splits), batch_size), desc="Embedding Progress"):
        batch = splits[i:i + batch_size]
        vectorstore.add_documents(batch)
        
    print("\n Vector Database Created and Persisted to disk.")
    return vectorstore

def ask_question(vectorstore, question):
    # AUTOMATIC HARDWARE DETECTION
    current_os = platform.system()
    
    if current_os == "Windows":
        print("DETECTED WINDOWS (NVIDIA MODE): Using Ultra Settings")
        CTX_SIZE = 8192 
        RETRIEVAL_K = 25
    else:
        # MAC SETTINGS
        print("DETECTED MAC : Using Light Settings")
        CTX_SIZE = 2048  # Reduced memory load
        RETRIEVAL_K = 5  # Reduced retrieval load

    print(f"Asking: {question}")
    print("Streaming Answer (Watch below)\n")
    
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model="llama3", 
        temperature=0,
        num_ctx=CTX_SIZE,
        base_url=OLLAMA_URL
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a precise technical analyst. Answer the question based ONLY on the context provided.
    
    Rules:
    1. If the answer is not in the context, state "I cannot find this information in the document."
    2. Be comprehensive but concise.
    
    Context:
    {context}
    
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,             
            "fetch_k": RETRIEVAL_K * 2,       
            "lambda_mult": 0.7 
        }
    )
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # STREAMING: Instead of .invoke(), use .stream()
    # This prints tokens one by one as they arrive
    full_answer = ""
    for chunk in retrieval_chain.stream({"input": question}):
        if 'answer' in chunk:
            print(chunk['answer'], end="", flush=True)
            full_answer += chunk['answer']
            
    print("\n\n--- Done ---")
    return full_answer

if __name__ == "__main__":
    test_pdf = os.path.join(BASE_DIR, "test_data.pdf")
    
    if os.path.exists(test_pdf):
        splits = ingest_document(test_pdf) 
        db = create_vector_db(splits)

        # Don't need to print response again,  streamed it
        ask_question(db, "Summarize the document.")
    else:
        print(f"Please add a file named 'test_data.pdf' to: {BASE_DIR}")