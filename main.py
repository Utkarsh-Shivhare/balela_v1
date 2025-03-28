from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from local_openai_chatbot import LocalOpenAIChatBot
import os
import json
from PyPDF2 import PdfReader
from typing import List
import asyncio
from pathlib import Path
import shutil
import uvicorn
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # Default placeholder for documentation

# Check if API key is available
if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
    logger.warning("No valid API key found. Please set your OPENAI_API_KEY in the .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="Homework Chatbot API", 
    description="API for uploading documents and chatting with OpenAI",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get current directory
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"Base directory: {BASE_DIR}")

# Configure upload settings
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize chatbot with API key
chatbot = LocalOpenAIChatBot(OPENAI_API_KEY)

# Helper functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

# API Routes
@app.get("/")
async def root():
    """API root endpoint with usage information"""
    return {
        "name": "Homework Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "POST - Upload documents with user_ids and document_ids",
            "/chat": "POST - Chat with the uploaded documents"
        },
        "status": "operational"
    }

@app.post("/upload/")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_ids: str = Form(...),
    document_ids: str = Form(...)
):
    """
    Handles the upload and processing of multiple files, associates them with 
    user IDs and document IDs, and processes their content.
    
    - **files**: The files to upload (PDF, JPG, PNG supported)
    - **user_ids**: Comma-separated list of user IDs, one per file
    - **document_ids**: Comma-separated list of document IDs, one per file
    """
    logger.info(f"Upload endpoint called with {len(files)} files")
    allowed_content_types = ["application/pdf", "image/jpeg", "image/png"]
    results = []
    user_ids_list = user_ids.split(",")
    doc_ids_list = document_ids.split(",")

    # Check if the number of user_ids and document_ids matches the number of files
    if len(files) != len(user_ids_list) or len(files) != len(doc_ids_list):
        raise HTTPException(
            status_code=400,
            detail="The number of files must match the number of user IDs and document IDs."
        )

    # Process each file
    for i in range(len(files)):
        file = files[i]
        user_id = user_ids_list[i]
        document_id = doc_ids_list[i]

        logger.info(f"Processing file: {file.filename}, content_type: {file.content_type}")
        
        # Some browsers might not set content_type correctly
        if not file.content_type or file.content_type == "application/octet-stream":
            # Guess content type from file extension
            ext = file.filename.split('.')[-1].lower()
            if ext in ['pdf']:
                file.content_type = "application/pdf"
            elif ext in ['jpg', 'jpeg']:
                file.content_type = "image/jpeg"
            elif ext in ['png']:
                file.content_type = "image/png"
            logger.info(f"Assigned content type {file.content_type} based on extension")

        if file.content_type not in allowed_content_types:
            error_msg = f"File '{file.filename}' is not allowed. Only PDF and image files are supported."
            logger.error(error_msg)
            results.append({"filename": file.filename, "error": error_msg})
            continue

        # Process the file
        try:
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract text from PDF
            if file.content_type == "application/pdf":
                extracted_data = extract_text_from_pdf(file_path)
            else:
                # For now, we're only handling PDFs in this implementation
                extracted_data = f"Image content from {file.filename} - text extraction not implemented"
            
            # Save to vector store
            if extracted_data:
                success = chatbot.save_document(extracted_data, user_id, document_id, False)
                if success:
                    results.append({"filename": file.filename, "status": "added to vector store"})
                else:
                    results.append({"filename": file.filename, "error": "Failed to save to vector store"})
            else:
                results.append({"filename": file.filename, "error": "Processing failed."})
            
            # Clean up
            os.remove(file_path)
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            results.append({"filename": file.filename, "error": f"Error processing file: {str(e)}"})

    return {"files_processed": len(results), "results": results}

@app.post("/chat/")
async def chat(
    query: str = Form(...),
    chat_history: str = Form("[]"),
    document_id: str = Form(...),
    user_id: str = Form(...)
):
    """
    Provides a chat interface that responds to queries using context from stored documents.
    
    - **query**: The user's question or prompt
    - **chat_history**: JSON string containing previous conversation (default empty array)
    - **document_id**: The ID of the document to search for context
    - **user_id**: The ID of the user who owns the document
    
    Returns a streaming response with chunks of the assistant's reply.
    """
    logger.info(f"Chat endpoint called for user_id: {user_id}, document_id: {document_id}")
    try:
        # Parse chat history
        chat_history_obj = json.loads(chat_history) if chat_history else []
        
        # Get context resources
        sources_formatted = chatbot.get_context_resources_from_db(query, user_id, document_id)
        
        # Return a StreamingResponse with the generator directly
        return StreamingResponse(
            chatbot.generate_response_stream(query, sources_formatted, chat_history_obj),
            media_type='text/event-stream'
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting API-only server on port {port}")
    try:
        # Use host 0.0.0.0 to listen on all interfaces
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        # Try alternate port if 8000 is unavailable
        try:
            port = 8080
            logger.info(f"Trying alternate port {port}...")
            uvicorn.run(app, host="0.0.0.0", port=port)
        except Exception as e2:
            logger.error(f"Error starting server on alternate port: {str(e2)}") 