# Balela AI

Balela AI is a comprehensive AI-powered learning assistant that helps users with homework, document analysis, and content generation.

## Features

- **Document Upload and Analysis**: Upload PDF documents and analyze their content
- **Conversational AI**: Interactive chat with AI that can answer questions about uploaded documents
- **Learning Assistance**: Generate questions, answers, and test analysis based on your materials
- **Writing Assistance**: Grammar checking, rephrasing, feedback, and more
- **Multimodal Chat**: Send images along with your text queries to get visual analysis and responses

## Authentication

All API endpoints (except the root `/` endpoint) require authentication. You must include the API key in your request headers using one of the following header names:

- `Authorization`
- `X-API-Key` 
- `api-key`

**Required API Key**: `a20d4fdd36062aa2bd56b6ee8b92cd1a`

If the API key is missing or incorrect, you'll receive a 400 status code with an authentication error message.

## New Feature: Multimodal Chat

You can now include images in your conversations with Balela AI. The AI will analyze the images and respond to your queries about them.

### How to Use Multimodal Chat

1. Use the `/chat/` endpoint with your regular query parameters
2. Add one or more image files using the `images` parameter in your form data
3. The AI will process both your text query and the images, providing a comprehensive response

Example using cURL:

```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Authorization: a20d4fdd36062aa2bd56b6ee8b92cd1a" \
  -F "query=What can you see in this image?" \
  -F "chat_history=[]" \
  -F "document_id=your_document_id" \
  -F "user_id=your_user_id" \
  -F "images=@/path/to/your/image.jpg"
```

Example using Python:

```python
import requests
import json

headers = {
    'Authorization': 'a20d4fdd36062aa2bd56b6ee8b92cd1a'
}

files = {
    'images': ('image.jpg', open('/path/to/your/image.jpg', 'rb'), 'image/jpeg')
}

data = {
    'query': 'What can you see in this image?',
    'chat_history': json.dumps([]),
    'document_id': 'your_document_id',
    'user_id': 'your_user_id'
}

response = requests.post('http://localhost:8000/chat/', 
                        headers=headers, 
                        data=data, 
                        files=files, 
                        stream=True)

# Process streaming response
for line in response.iter_lines():
    if line:
        # The response is in the format: data: {"chunk": "text_chunk"}
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_json = json.loads(line_str[6:])
            if 'chunk' in data_json:
                print(data_json['chunk'], end='')
```

### Technical Details

- Supports JPEG, PNG, and other common image formats
- Uses GPT-4o model with multimodal capabilities
- Combines document context with image analysis for comprehensive responses
- Streaming responses for real-time feedback

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables (see `.env.example`)
4. Run the server:
   ```
   uvicorn main:app --reload
   ```

## API Endpoints

All endpoints require authentication headers as described above.

- `/upload/` - Upload documents for vectorization
- `/chat/` - Interactive chat with context from documents and optional images
- `/delete_document/` - Remove documents from the system
- Additional endpoints for learning and writing assistance features


sudo nano /etc/nginx/sites-available/balela

server {
    listen 80;
    server_name 209.97.187.195;  # Replace with your domain if available

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}

## License

[License Information] 
