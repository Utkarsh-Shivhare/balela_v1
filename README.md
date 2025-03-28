# Balela V1 - AI Homework Chatbot

A Flask-based API that allows users to upload documents and chat with them using OpenAI's GPT models.

## Features

- Document upload (PDF, JPG, PNG) with vector store for retrieval
- Streaming chat responses using GPT-4o for intelligent homework help
- Simple API design with two main endpoints: `/upload` and `/chat`
- Flask-based server with CORS support

## Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/balela_v1.git
cd balela_v1
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```
cp .env.example .env
```
Then edit the `.env` file and add your OpenAI API key.

4. Run the application:
```
python flask_app.py
```

The server will start on port 8000 by default (with a fallback to port 8080).

## API Endpoints

### Root (`/`)
- **Method**: GET
- **Description**: Returns basic API information and available endpoints
- **Response**: JSON with API name, version, endpoints, and status

### Upload (`/upload/`)
- **Method**: POST
- **Description**: Upload documents with user IDs and document IDs
- **Parameters**:
  - `files`: The files to upload (PDF, JPG, PNG supported)
  - `user_ids`: Comma-separated list of user IDs, one per file
  - `document_ids`: Comma-separated list of document IDs, one per file
- **Response**: JSON with upload results

### Chat (`/chat/`)
- **Method**: POST
- **Description**: Chat with uploaded documents
- **Parameters**:
  - `query`: The user's question or prompt
  - `chat_history`: JSON string containing previous conversation (default empty array)
  - `document_id`: The ID of the document to search for context
  - `user_id`: The ID of the user who owns the document
- **Response**: Streaming response with chunks of the assistant's reply

## Implementation Details

- Uses LangChain with OpenAI for embeddings and chat completions
- Stores document embeddings locally in vector_store directory
- Provides educational assistance following the "guide don't solve" principle

## License

MIT 