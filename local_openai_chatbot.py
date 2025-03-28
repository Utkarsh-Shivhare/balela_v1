# Import libraries
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import logging
from langchain_core.output_parsers import StrOutputParser
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LocalOpenAIChatBot:
    def __init__(self, OPENAI_API_KEY):
        load_dotenv()
        
        logging.info("Initializing LocalOpenAIChatBot")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=OPENAI_API_KEY,
            streaming=True
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("vector_store")
        self.data_dir.mkdir(exist_ok=True)
        
        # Use the same template as the homework chatbot
        from instruction.home_work_chatbot_template import HOMEWORK_CHATBOT_TEMPLATE
        self.GROUNDED_PROMPT = HOMEWORK_CHATBOT_TEMPLATE
        self.prompt = PromptTemplate(
            template=self.GROUNDED_PROMPT,
            input_variables=["query", "sources", "chat_history"]
        )

    def save_document(self, content, user_id, document_id, is_book=False):
        """Save a document with its embedding to the local store."""
        try:
            # Generate embedding for the content
            embedding = self.embeddings.embed_query(content)
            
            # Create document entry
            document = {
                "content": content,
                "embedding": embedding,
                "user_id": user_id,
                "document_id": document_id,
                "is_book": is_book,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create filename based on user_id and document_id
            filename = self.data_dir / f"{user_id}_{document_id}.json"
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(document, f)
            
            logging.info(f"Document saved successfully to {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving document: {str(e)}")
            return False

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query, user_id, document_id):
        logging.info(f"Searching for query: '{query}' for user_id: '{user_id}' and document_id: '{document_id}'")
        try:
            # Get embeddings for the query
            query_embedding = self.embeddings.embed_query(query)
            
            results = []
            # Load and search through all documents for the specific user and document
            filename = self.data_dir / f"{user_id}_{document_id}.json"
            
            if not filename.exists():
                logging.warning(f"No documents found for user_id: {user_id} and document_id: {document_id}")
                return []
            
            with open(filename, 'r') as f:
                doc = json.load(f)
                
            if not doc["is_book"]:  # Skip if it's marked as a book
                similarity = self.cosine_similarity(query_embedding, doc["embedding"])
                results.append({
                    "metadata": {"content": doc["content"]},
                    "score": similarity
                })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            raise

    def format_sources(self, search_results):
        logging.info("Formatting search results.")
        formatted_sources = "=================\n".join([
            f'CONTENT: {result["metadata"]["content"]}' 
            for result in search_results
        ])
        logging.debug(f"Formatted sources: {formatted_sources}")
        return formatted_sources

    def generate_response_stream(self, query, sources_formatted, chat_history):
        logging.info("Streaming response generation using Prompt Template.")
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            
            response = chain.stream({
                "query": query,
                "sources": sources_formatted,
                "chat_history": chat_history,
            })
            for chunk in response:
                yield chunk

        except Exception as e:
            logging.error(f"Error during streaming generation: {str(e)}")
            yield json.dumps({"error": str(e)})

    def get_context_resources_from_db(self, query, user_id, document_id):
        logging.info(f"Starting chat with query: '{query}'")
        search_results = self.search(query, user_id, document_id)
        sources_formatted = self.format_sources(search_results)
        logging.debug(f"Sources formatted: {len(sources_formatted)} characters")
        return sources_formatted 