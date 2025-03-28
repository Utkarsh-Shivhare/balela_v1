from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from local_openai_chatbot import LocalOpenAIChatBot
import os
import json
from PyPDF2 import PdfReader
import time
from pathlib import Path
import shutil
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

# Initialize Flask app
app = Flask(__name__)

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

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# Routes
@app.route('/', methods=['GET'])
def root():
    """API root endpoint with usage information"""
    return jsonify({
        "name": "Homework Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "POST - Upload documents with user_ids and document_ids",
            "/chat": "POST - Chat with the uploaded documents"
        },
        "status": "operational"
    })

@app.route('/upload/', methods=['POST'])
def upload_documents():
    """
    Handles the upload and processing of multiple files, associates them with 
    user IDs and document IDs, and processes their content.
    """
    logger.info("Upload endpoint called")
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files')
    user_ids = request.form.get('user_ids', '')
    document_ids = request.form.get('document_ids', '')
    
    if not files or not user_ids or not document_ids:
        return jsonify({"error": "Missing required parameters"}), 400
    
    allowed_content_types = ["application/pdf", "image/jpeg", "image/png"]
    results = []
    user_ids_list = user_ids.split(",")
    doc_ids_list = document_ids.split(",")

    # Check if the number of user_ids and document_ids matches the number of files
    if len(files) != len(user_ids_list) or len(files) != len(doc_ids_list):
        return jsonify({
            "error": "The number of files must match the number of user IDs and document IDs."
        }), 400

    # Process each file
    for i in range(len(files)):
        file = files[i]
        user_id = user_ids_list[i]
        document_id = doc_ids_list[i]

        if file.filename == '':
            results.append({"error": "Empty filename"})
            continue

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
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
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

    return jsonify({"files_processed": len(results), "results": results})

@app.route('/chat/', methods=['POST'])
def chat():
    """
    Provides a chat interface that responds to queries using context from stored documents.
    """
    try:
        query = request.form.get('query', '')
        chat_history = request.form.get('chat_history', '[]')
        document_id = request.form.get('document_id', '')
        user_id = request.form.get('user_id', '')
        
        if not query or not document_id or not user_id:
            return jsonify({"error": "Missing required parameters"}), 400
        
        logger.info(f"Chat endpoint called for user_id: {user_id}, document_id: {document_id}")
        
        # Parse chat history
        chat_history_obj = json.loads(chat_history) if chat_history else []
        
        # Get context resources
        sources_formatted = chatbot.get_context_resources_from_db(query, user_id, document_id)
        
        # Create streaming response
        def generate():
            try:
                for chunk in chatbot.generate_response_stream(query, sources_formatted, chat_history_obj):
                    yield chunk
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream'
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Flask server on port {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        # Try alternate port if 8000 is unavailable
        try:
            port = 8080
            logger.info(f"Trying alternate port {port}...")
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        except Exception as e2:
            logger.error(f"Error starting server on alternate port: {str(e2)}") 