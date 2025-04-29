# Balela AI

Balela AI is a comprehensive AI-powered learning assistant that helps users with homework, document analysis, and content generation.

## Features

- **Document Upload and Analysis**: Upload PDF documents and analyze their content
- **Conversational AI**: Interactive chat with AI that can answer questions about uploaded documents
- **Learning Assistance**: Generate questions, answers, and test analysis based on your materials
- **Writing Assistance**: Grammar checking, rephrasing, feedback, and more
- **Multimodal Chat**: Send images along with your text queries to get visual analysis and responses

## New Feature: Multimodal Chat

You can now include images in your conversations with Balela AI. The AI will analyze the images and respond to your queries about them.

### How to Use Multimodal Chat

1. Use the `/chat/` endpoint with your regular query parameters
2. Add one or more image files using the `images` parameter in your form data
3. The AI will process both your text query and the images, providing a comprehensive response

Example using cURL:

```bash
curl -X POST "http://localhost:8000/chat/" \
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

files = {
    'images': ('image.jpg', open('/path/to/your/image.jpg', 'rb'), 'image/jpeg')
}

data = {
    'query': 'What can you see in this image?',
    'chat_history': json.dumps([]),
    'document_id': 'your_document_id',
    'user_id': 'your_user_id'
}

response = requests.post('http://localhost:8000/chat/', data=data, files=files, stream=True)

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

- `/upload/` - Upload documents for vectorization
- `/chat/` - Interactive chat with context from documents and optional images
- `/delete_document/` - Remove documents from the system
- Additional endpoints for learning and writing assistance features

## License

[License Information] 