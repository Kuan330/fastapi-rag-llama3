import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_engine import ingest_document, create_vector_db, ask_question, DB_DIR

# Initialize the App
app = FastAPI(
    title="Local RAG Engine",
    description="A production-ready API for chatting with private documents using Llama-3.",
    version="1.0.0"
)

# Global variable to hold the database connection in memory
# This avoids reloading the heavy database for every single request.
VECTOR_DB = None

# Define the data model for the input query
class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_existing_db():
    """
    On server startup, try to load the existing database if it exists.
    """
    global VECTOR_DB
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    
    if os.path.exists(DB_DIR):
        print("Loading existing Vector DB from disk")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        VECTOR_DB = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        print("DB Loaded Successfully")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global VECTOR_DB
    
    # FIX: Force release the database connection before try to delete the folder
    VECTOR_DB = None 
    
    file_location = f"temp_{file.filename}"
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        splits = ingest_document(file_location)
        VECTOR_DB = create_vector_db(splits)
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks": len(splits),
            "message": "Document processed and AI is ready."
        }
        
    except Exception as e:
        # This prints the REAL error to the terminal to visualise it
        print(f"CRITICAL ERROR: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
            
@app.post("/query")
async def query_ai(request: QueryRequest):
    """
    Asks a question to the loaded document.
    """
    global VECTOR_DB
    
    if VECTOR_DB is None:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a PDF first.")
    
    try:
        # Get the answer from Llama-3
        answer = ask_question(VECTOR_DB, request.question)
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint just to check if server is alive
@app.get("/")
async def root():
    return {"message": "RAG Engine is Online. Go to /docs to use the interface."}