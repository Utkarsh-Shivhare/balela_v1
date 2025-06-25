from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from database import delete_document_by_document_id
from learning_assistant_agents import *
from local_openai_chatbot import *
from vector_store import WeaviateVectorStore
from typing import List, Optional
from pydantic import BaseModel
import os
import json
from PyPDF2 import PdfReader
import pymupdf
from typing import List
import asyncio
from pathlib import Path
import shutil
import uvicorn
import logging
from dotenv import load_dotenv
from writing_assistant_agents import *
from vector_store import *
import base64

# Load environment variables
load_dotenv()

# Authentication configuration
REQUIRED_API_KEY = "a20d4fdd36062aa2bd56b6ee8b92cd1a"

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str
class Test_Analyzer_Request(BaseModel):
    user_id: str
    document_id: str
    qa_pairs: List[QuestionAnswerPair]
class Rephrase_Support_Request(BaseModel):
    content_to_be_rephrased: str
    user_query: Optional[str] = None
class AssignmentInput(BaseModel):
    title_of_assignment: str
    assignment_text: str
    questions: Optional[List[str]] = None
class FeedbackRequestBasedOnStrictness(BaseModel):
    input_content_text: str
    strictness_level: str

class GrammarRequest(BaseModel):
    user_input: str

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

# Authentication middleware
@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """
    Middleware to check for authentication key in headers
    """
    # Skip authentication for root endpoint (for health checks)
    if request.url.path == "/":
        response = await call_next(request)
        return response
    
    # Check for authentication header
    auth_key = request.headers.get("Authorization") or request.headers.get("X-API-Key") or request.headers.get("api-key")
    
    if not auth_key or auth_key != REQUIRED_API_KEY:
        logger.warning(f"Unauthorized access attempt from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "Authentication required",
                "message": "Valid API key must be provided in headers (Authorization, X-API-Key, or api-key)"
            }
        )
    
    # If authentication passes, continue with the request
    response = await call_next(request)
    return response

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*","https://balela.co.za/backend","https://balela.co.za/","*"],
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

# Initialize vector store
vector_store = WeaviateVectorStore()

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

@app.post("/upload/", tags=["Homework Assistance"])
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
                print("extracted_data", extracted_data)
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

@app.post("/chat/", tags=["Homework Assistance"])
async def chat(
    query: str = Form(...),
    chat_history: str = Form("[]"),
    document_id: str = Form(...),
    user_id: str = Form(...),
    images: List[UploadFile] = File([])  # Make it optional
):
    """
    Provides a chat interface that responds to queries using context from stored documents.
    
    - **query**: The user's question or prompt
    - **chat_history**: JSON string containing previous conversation (default empty array)
    - **document_id**: The ID of the document to search for context
    - **user_id**: The ID of the user who owns the document
    - **images**: Optional list of image files to include in the conversation
    
    Returns a streaming response with chunks of the assistant's reply.
    """
    logger.info(f"Chat endpoint called for user_id: {user_id}, document_id: {document_id}, with {len(images)} images")
    try:
        # Parse and validate chat_history
        try:
            chat_history_obj = json.loads(chat_history)
            if not isinstance(chat_history_obj, list):
                raise ValueError("chat_history must be a list")
        except Exception as e:
            logger.warning(f"Invalid chat_history format: {chat_history}. Error: {e}")
            chat_history_obj = []

        # Get relevant context (optional logic; can be removed if not needed)
        sources_formatted = chatbot.get_context_resources_from_db(query, user_id, document_id)
        if not sources_formatted:
            sources_formatted = ""

        # Read and encode images
        image_data = []
        for img in images:
            logger.info(f"Received image: {img.filename}")
            img_content = await img.read()
            base64_img = base64.b64encode(img_content).decode("utf-8")
            mime_type = img.content_type or "image/jpeg"
            image_data.append({
                "base64_data": base64_img,
                "mime_type": mime_type
            })

        # Stream response back to the client
        return StreamingResponse(
            chatbot.generate_response_stream(query, sources_formatted, chat_history_obj, image_data),
            media_type='text/event-stream'
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/upload_book/", tags=['Learning Assistant'])
async def upload_book(
    files: List[UploadFile] = File(...),
    user_ids: str = Form(...),
    document_ids: str = Form(...)
):
    """
    Processes uploaded files (PDF, JPEG, PNG) to extract textual data, associates
    the data with user and document IDs, and stores it in a vector store with the book flag.
    
    - **files**: The files to upload (PDF, JPG, PNG supported)
    - **user_ids**: Comma-separated list of user IDs, one per file
    - **document_ids**: Comma-separated list of document IDs, one per file
    """
    logger.info(f"Upload book endpoint called with {len(files)} files")
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

        logger.info(f"Processing book file: {file.filename}, content_type: {file.content_type}")
        
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
            
            # Save to vector store with is_book flag set to True
            if extracted_data:
                success = chatbot.save_document(extracted_data, user_id, document_id, True)  # Mark as book
                if success:
                    results.append({"filename": file.filename, "status": "added to vector store as book"})
                    logger.info(f"Book '{file.filename}' uploaded successfully.")
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

# Learning Assistant APIs - Placeholder implementations
@app.post("/analyze_content/", tags=['Learning Assistant'])
async def analyze_uploaded_material(
    user_id: str = Form(...),
    document_id: str = Form(...)          
):
    """
    Analyzes a specific document uploaded by a user to generate insights or 
    learning content.

    Inputs:
    - OPENAI_API_KEY (str): The OpenAI API key for authentication.
    - user_id (str): ID of the user requesting the analysis.
    - document_id (str): ID of the document to analyze.

    Returns:
    JSON response:
    - status (str): `"success"` if the analysis was completed.
    - generated_questions (list): Questions generated based on the document content.
    """
    try:
        if user_id and document_id:
            learning_assistant_instance = LearningAssistant(OPENAI_API_KEY)
            response = learning_assistant_instance.call_analyze_content_agent(user_id, document_id)
            logger.info(f"Analysis completed for user_id '{user_id}' and document_id '{document_id}'.")
            return {
                "status": "success",
                "generated_questions": response,
            }
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )

@app.post("/question_generation/", tags=['Learning Assistant'])
async def question_generator_assistant(
    user_id: str = Form(...),
    document_id: str = Form(...)        
):
    """
    Generates contextually relevant questions based on the content of a specific 
    document associated with a user.

    Inputs:
    - OPENAI_API_KEY (str): The OpenAI API key for authentication.
    - user_id (str): ID of the user requesting question generation.
    - document_id (str): ID of the document to generate questions from.

    Returns:
    JSON response:
    - status (str): `"success"` if question generation was successful.
    - generated_questions (list): List of questions generated from the document content.
    """
    try:
        if user_id and document_id:
            learning_assistant_instance = LearningAssistant(OPENAI_API_KEY)
            response = learning_assistant_instance.call_generate_questions_agent(user_id, document_id)
            logger.info(f"Question generation completed for user_id '{user_id}' and document_id '{document_id}'.")
            return {
                "status": "success",
                "generated_questions": response,
            }
    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Question generation failed: {str(e)}"}
        )

@app.post("/guided_question_assistant/", tags=['Learning Assistant'])
async def guided_question_assistant(
    user_id: str = Form(...),
    document_id: str = Form(...),
    question_query: str = Form(...),
    user_input_answer: str = Form(...)                
):
    """
    Provides assistance with guided questions based on a document and user-provided
    input. Validates and responds to user input using document content.

    Inputs:
    - OPENAI_API_KEY (str): The OpenAI API key for authentication.
    - user_id (str): ID of the user.
    - document_id (str): ID of the document.
    - question_query (str): The question posed by the user.
    - user_input_answer (str): User's input answer to the question.

    Returns:
    JSON response:
    - status (str): `"success"` if the guided question assistance was successful.
    - generated_questions (list): Relevant questions or guidance based on the user input.
    """
    try:
        if user_id and document_id:
            learning_assistant_instance = LearningAssistant(OPENAI_API_KEY)
            response = learning_assistant_instance.call_guided_question_assistant(user_id, document_id, question_query, user_input_answer)
            logger.info(f"Guided question assistance completed for user_id '{user_id}' and document_id '{document_id}'.")
            return {
                "status": "success",
                "generated_questions": response,
            }
    except Exception as e:
        logger.error(f"Guided question assistance failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Guided question assistance failed: {str(e)}"}
        )

@app.post("/answer_generation_assistant/", tags=['Learning Assistant'])
async def answer_generation_assistant(
    user_id: str = Form(...),
    document_id: str = Form(...),
    question_input: str = Form(...)                   
):
    """
    Generates a detailed answer to a question based on the content of a specific
    document associated with a user.

    Inputs:
    - OPENAI_API_KEY (str): The OpenAI API key for authentication.
    - user_id (str): ID of the user.
    - document_id (str): ID of the document.
    - question_input (str): The question to generate an answer for.

    Returns:
    JSON response:
    - status (str): `"success"` if the answer generation was successful.
    - generated_answer (str): Detailed answer generated from the document content.
    """
    try:
        if user_id and document_id:
            learning_assistant_instance = LearningAssistant(OPENAI_API_KEY)
            response = learning_assistant_instance.call_detailed_answer_agent(user_id, document_id, question_input)
            logger.info(f"Answer generation completed for user_id '{user_id}' and document_id '{document_id}'.")
            return {
                "status": "success",
                "generated_answer": response,
            }
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Answer generation failed: {str(e)}"}
        )

@app.delete("/delete_document/", tags=["Homework Assistance"])
async def delete_document_endpoint(
    document_id: str = Form(...)
):
    """
    Delete a document and its associated vector embeddings.
    """
    try:
        # Delete from vector store
        vector_store_success = vector_store.delete_documents(document_id)
        
        # Delete from SQL database (if needed)
        # sql_success = delete_documents(document_id)
        
        if vector_store_success:
            return {"message": f"Document {document_id} successfully deleted"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document {document_id} from one or more storage systems"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@app.post("/test_analyser/", tags=['Learning Assistant'])
async def test_and_exam_analyser(request: Test_Analyzer_Request):
    """
    Submits multiple question-answer pairs for analysis, providing insights 
    or feedback based on the test content.

    Inputs:
    - request (Test_Analyzer_Request): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - user_id (str): ID of the user.
        - document_id (str): ID of the document.
        - qa_pairs (list): List of question-answer pairs.

    Returns:
    JSON response:
    - status (str): `"success"` if analysis is successful.
    - test_analysis (list): Analysis results of the question-answer pairs.
    """
    try:
        # Extract data from the request
        user_id = request.user_id
        document_id = request.document_id
        qa_pairs = request.qa_pairs

        if user_id and document_id:
            learning_assistant_instance = LearningAssistant(OPENAI_API_KEY)
            response = learning_assistant_instance.call_analyze_test(user_id, document_id, qa_pairs)
            logger.info(f"Test analysis completed for user_id '{user_id}' and document_id '{document_id}'.")
            return {
                "status": "success",
                "test_analysis": response
            }
        else:
            logger.warning("User ID or Document ID is missing.")
            return {
                "status": "error",
                "message": "Please check user_id and document_id."
            }
    except Exception as e:
        logger.error(f"Failed to process QA pairs: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process QA pairs: {str(e)}"}
        )

# Writing Assistant APIs - Placeholder implementations
@app.post("/grammer_and_language_support/", tags=['Writing Assistant'])
async def grammer_and_language_support_writer(data: GrammarRequest):
    """
    Processes input text for grammar corrections and language improvements.

    Inputs:
    - input (Grammer_Language_Support_Request): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - user_input (str): The text provided by the user.

    Returns:
    JSON response:
    - status (str): `"success"` if processing is successful.
    - response (str): Corrected text with grammar and language improvements.
    """
    user_input = data.user_input
    try:
        writing_assistant_instance = WritingAssistant(OPENAI_API_KEY)
        corrected_response = writing_assistant_instance.call_grammar_and_language_support_agent(user_input)
        logger.info("Grammar and language support processed successfully.")
        return {"status": "success", "response": corrected_response}
    except Exception as e:
        logger.error(f"Grammar and language support failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Grammar and language support failed: {str(e)}"}
        )

@app.post("/rephrase_text_support/", tags=['Writing Assistant'])
async def rephrase_text_support_writer(input: Rephrase_Support_Request):
    """
    Rephrases input text based on user-provided content and context.

    Inputs:
    - input (Rephrase_Support_Request): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - content_to_be_rephrased (str): Text to be rephrased.
        - user_query (str): Additional context or query for rephrasing.

    Returns:
    JSON response:
    - status (str): `"success"` if rephrasing is successful.
    - response (str): Rephrased text.
    """
    try:
        content_to_be_rephrased = input.content_to_be_rephrased  
        user_query = input.user_query
        
        writing_assistant_instance = WritingAssistant(OPENAI_API_KEY)
        rephrased_response = writing_assistant_instance.call_rephrasing_agent(content_to_be_rephrased, user_query)
        
        logger.info("Rephrasing support processed successfully.")
        return {"status": "success", "response": rephrased_response}
    except Exception as e:
        logger.error(f"Rephrasing support failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Rephrasing support failed: {str(e)}"}
        )

@app.post("/assignment_feedback_support/", tags=['Writing Assistant'])
async def assignment_feedback(input: AssignmentInput):
    """
    Provides detailed feedback on the text of an assignment.

    Inputs:
    - input (AssignmentInput): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - assignment_text (str): Text of the assignment for feedback.
        - title (str): Title of the assignment for context in feedback.

    Returns:
    JSON response:
    - status (str): `"success"` if feedback generation is successful.
    - response (str): Feedback on the assignment.
    """
    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        response = writing_assistant.call_assignment_feedback_agent(input.assignment_text,input.title_of_assignment,input.questions)
        logger.info("Assignment feedback processed successfully.")
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Assignment feedback failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Assignment feedback failed: {str(e)}"}
        )

@app.post("/custom_feedback_using_strictness_level_support/", tags=['Writing Assistant'])
async def provide_feedback_based_on_strictness(input: FeedbackRequestBasedOnStrictness):
    """
    Generates assignment feedback based on the specified strictness level.

    Inputs:
    - input (FeedbackRequestBasedOnStrictness): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - input_content_text (str): Text for feedback generation.
        - strictness_level (int): Strictness level for feedback.

    Returns:
    JSON response:
    - status (str): `"success"` if feedback generation is successful.
    - response (str): Feedback content based on the strictness level.
    """
    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        response = writing_assistant.call_custom_feedback_agent_with_strictness_level(input.input_content_text, input.strictness_level)
        logger.info("Custom feedback processed successfully.")
        return {"status": "success", "response": response}  # Return the ai_comment field
    except Exception as e:
        logger.error(f"Custom feedback generation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Custom feedback generation failed: {str(e)}"}
        )

@app.post("/structural_flow_analysis_support/", tags=['Writing Assistant'])
async def structural_flow_analysis(user_content: str):
    """
    Analyzes the structure and flow of the provided content for logical and 
    stylistic improvements.

    Inputs:
    - input (AnalysisRequest): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - user_content (str): Content to analyze for structure and flow.

    Returns:
    JSON response:
    - status (str): `"success"` if analysis is successful.
    - response (str): Analysis of the content's structure and flow.
    """

    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        response = writing_assistant.call_structural_flow_analysis(user_content)
        logger.info("Structural flow analysis processed successfully.")
        return {
            "status": "success",
            "response":response
        }
    except Exception as e:
        logger.error(f"Structural flow analysis failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Structural flow analysis failed: {str(e)}"}
        )

@app.post("/content_relevance_check/", tags=['Writing Assistant'])
async def content_relevance_check(user_content: str):
    """
    Evaluates the relevance of the provided content against a given context.

    Inputs:
    - input (AnalysisRequest): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - user_content (str): Content to check for relevance.

    Returns:
    JSON response:
    - status (str): `"success"` if relevance check is successful.
    - response (str): Relevance evaluation of the content.
    """
    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        response = writing_assistant.call_content_relevance_check(user_content)
        logger.info("Content relevance check processed successfully.")
        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        logger.error(f"Content relevance check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Content relevance check failed: {str(e)}"}
        )

@app.post("/encouraging_revisions/", tags=['Writing Assistant'])
async def encouraging_revisions(user_content: str):
    """
    Encourages and suggests refinements in user-provided content to improve 
    quality and clarity.

    Inputs:
    - input (AnalysisRequest): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - user_content (str): Content to refine and improve.

    Returns:
    JSON response:
    - status (str): `"success"` if suggestions are generated successfully.
    - response (str): Suggestions for content improvement.
    """
    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        response = writing_assistant.call_encouraging_revisions(user_content)
        logger.info("Encouraging revisions processed successfully.")
        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        logger.error(f"Encouraging revisions failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Encouraging revisions failed: {str(e)}"}
        )

@app.post("/research_project_assistance/", tags=['Writing Assistant'])
async def research_project_assistance(user_content: str):
    """
    Assists with structuring and refining research projects by analyzing 
    the provided content.

    Inputs:
    - input (AnalysisRequest): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - user_content (str): Content related to the research project.

    Returns:
    JSON response:
    - status (str): `"success"` if assistance is provided successfully.
    - response (str): Suggestions and guidance for the research project.
    """
    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        response = writing_assistant.call_research_project_assistance(user_content)
        logger.info("Research project assistance processed successfully.")
        return {
            "status": "success",
           "response":response
        }
    except Exception as e:
        logger.error(f"Research project assistance failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Research project assistance failed: {str(e)}"}
        )

@app.post("/autocomplete/", tags=["Writing Assistant"])
async def autocomplete(input_text: str):
    """
    Provides auto-complete suggestions based on the input text.

    Inputs:
    - input (AutoCompleteRequest): A request object containing:
        - OPENAI_API_KEY (str): The OpenAI API key for authentication.
        - input_text (str): The text for which suggestions are needed.

    Returns:
    JSON response:
    - status (str): `"success"` if suggestions are generated successfully.
    - suggestions (list): List of auto-complete suggestions.
    """
    try:
        writing_assistant = WritingAssistant(OPENAI_API_KEY)
        auto_complete_suggestion = writing_assistant.call_suggestion_auto_complete_agent(input_text)
        logger.info("Auto-complete suggestions generated successfully.")
        return {
            "status": "success",
            "suggestions": auto_complete_suggestion
        }
    except Exception as e:
        logger.error(f"Auto-complete failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Auto-complete failed: {str(e)}"}
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
