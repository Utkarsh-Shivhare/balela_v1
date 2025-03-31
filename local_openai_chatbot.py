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
from datetime import datetime
from database import init_db, save_document_to_db, get_document_from_db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LocalOpenAIChatBot:
    def __init__(self, OPENAI_API_KEY):
        # Load environment variables, useful for additional configuration
        load_dotenv()
        
        logger.info("Initializing LocalOpenAIChatBot")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=OPENAI_API_KEY,
            streaming=True
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Initialize database connection
        self.engine, Session = init_db()
        self.Session = Session
        
        # Import template with proper path handling to avoid import issues
        try:
            from instruction.home_work_chatbot_template import HOMEWORK_CHATBOT_TEMPLATE
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from instruction.home_work_chatbot_template import HOMEWORK_CHATBOT_TEMPLATE
            
        self.GROUNDED_PROMPT = HOMEWORK_CHATBOT_TEMPLATE
        self.prompt = PromptTemplate(
            template=self.GROUNDED_PROMPT,
            input_variables=["query", "sources", "chat_history"]
        )

    def save_document(self, content, user_id, document_id, is_book=False):
        """Save a document with its embedding to the database."""
        try:
            # Generate embedding for the content
            embedding = self.embeddings.embed_query(content)
            
            # Create database session
            session = self.Session()
            
            # Save to database
            success = save_document_to_db(
                session=session,
                user_id=user_id,
                document_id=document_id,
                content=content,
                embedding=embedding,
                is_book=is_book
            )
            
            session.close()
            return success
            
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            return False

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query, user_id, document_id):
        logger.info(f"Searching for query: '{query}' for user_id: '{user_id}' and document_id: '{document_id}'")
        try:
            # Get embeddings for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Create database session
            session = self.Session()
            
            # Get document from database
            doc = get_document_from_db(session, user_id, document_id)
            session.close()
            
            if not doc:
                logger.warning(f"No documents found for user_id: {user_id} and document_id: {document_id}")
                return []
            
            results = []
            if not doc["is_book"]:
                similarity = self.cosine_similarity(query_embedding, doc["embedding"])
                results.append({
                    "metadata": {"content": doc["content"]},
                    "score": similarity
                })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def format_sources(self, search_results):
        logger.info("Formatting search results.")
        formatted_sources = "=================\n".join([
            f'CONTENT: {result["metadata"]["content"]}' 
            for result in search_results
        ])
        logger.debug(f"Formatted sources: {len(formatted_sources)} characters")
        return formatted_sources

    def generate_response_stream(self, query, sources_formatted, chat_history):
        logger.info("Streaming response generation using Prompt Template.")
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
            logger.error(f"Error during streaming generation: {str(e)}")
            yield json.dumps({"error": str(e)})

    def get_context_resources_from_db(self, query, user_id, document_id):
        logger.info(f"Starting chat with query: '{query}'")
        search_results = self.search(query, user_id, document_id)
        sources_formatted = self.format_sources(search_results)
        logger.debug(f"Sources formatted: {len(sources_formatted)} characters")
        return sources_formatted 