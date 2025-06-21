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
from vector_store import WeaviateVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LocalOpenAIChatBot:
    def __init__(self, OPENAI_API_KEY):
        # Load environment variables, useful for additional configuration
        load_dotenv()
        
        logger.info("Initializing LocalOpenAIChatBot")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4o",  # Updated to gpt-4o for multimodal support
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=OPENAI_API_KEY,
            streaming=True
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Initialize Weaviate vector store
        self.vector_store = WeaviateVectorStore()
        # Add embeddings to vector store instance for search method
        self.vector_store.embeddings = self.embeddings
        
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
        """Save a document with its embedding to Weaviate."""
        try:
            logger.info(f"Attempting to save document for user_id: {user_id}, document_id: {document_id}")
            logger.debug(f"Content preview (first 100 chars): {content[:100]}...")
            
            # Clean and prepare content
            if content:
                content = content.strip()
            else:
                logger.error("Empty content provided")
                return False
                
            # Generate embedding for the content
            try:
                embedding = self.embeddings.embed_query(content)
                # Normalize the embedding before saving
                embedding = self.vector_store._normalize_vector(embedding)
                logger.info("Successfully generated and normalized embedding")
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                return False
            
            # Prepare metadata with additional information
            metadata = {
                "is_book": is_book,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content),
                "has_math": any(char in content for char in "+-*/^=()[]{}\\")
            }
            
            # Save to Weaviate with chunking for large content
            chunk_size = 2000  # Adjust based on your needs
            if len(content) > chunk_size:
                chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
                logger.info(f"Content split into {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                    
                    # Generate and normalize embedding for chunk
                    chunk_embedding = self.embeddings.embed_query(chunk)
                    chunk_embedding = self.vector_store._normalize_vector(chunk_embedding)
                    
                    success = self.vector_store.add_documents(
                        texts=[chunk],
                        embeddings=[chunk_embedding],
                        user_id=user_id,
                        document_id=f"{document_id}_chunk_{i}",
                        metadata=chunk_metadata
                    )
                    
                    if not success:
                        logger.error(f"Failed to save chunk {i}")
                        return False
                    
                logger.info("Successfully saved all chunks")
                return True
            else:
                # Save single document
                success = self.vector_store.add_documents(
                    texts=[content],
                    embeddings=[embedding],
                    user_id=user_id,
                    document_id=document_id,
                    metadata=metadata
                )
                
                if success:
                    logger.info("Successfully saved document")
                    return True
                else:
                    logger.error("Failed to save document")
                    return False
            
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            return False

    def search(self, query, user_id, document_id=None):
        """Search for documents using hybrid search approach."""
        logger.info(f"Searching for query: '{query}' for user_id: '{user_id}' and document_id: '{document_id}'")
        try:
            # Use the enhanced search method from WeaviateVectorStore
            results = self.vector_store.search(query, user_id, document_id)
            
            if not results:
                logger.warning(f"No documents found for user_id: {user_id} and document_id: {document_id}")
                return []
            
            # Log search results for debugging
            logger.info(f"Found {len(results)} relevant documents")
            for i, result in enumerate(results[:3]):  # Log first 3 results
                logger.debug(f"Result {i+1} score: {result['score']:.3f}")
                logger.debug(f"Result {i+1} preview: {result['metadata']['content'][:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []

    def format_sources(self, search_results):
        """Format search results for response generation with improved scoring information."""
        logger.info("Formatting search results.")
        
        if not search_results:
            return "No relevant sources found."
        
        # Format each result with enhanced score information
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            content = result["metadata"]["content"]
            score = result["score"]
            has_math = result["metadata"].get("has_math", False)
            
            # Add header with detailed relevance info
            header = f"SOURCE {i} (Relevance Score: {score:.3f})"
            if score >= 0.8:
                header += " [High Relevance]"
            elif score >= 0.65:
                header += " [Medium Relevance]"
            
            if has_math:
                header += " [Contains Mathematical Content]"
            
            # Format the content
            formatted_result = f"{header}\n{'-' * len(header)}\n{content}"
            formatted_results.append(formatted_result)
        
        # Join all formatted results with clear separation
        formatted_sources = "\n\n=================\n\n".join(formatted_results)
        
        logger.debug(f"Formatted sources: {len(formatted_sources)} characters")
        return formatted_sources

    def generate_response_stream(self, query, sources_formatted, chat_history, image_data=None):
        """Generate streaming response from LLM, with optional images."""
        logger.info("Streaming response generation using Prompt Template.")
        try:
            # If no images are provided: normal text prompt flow
            if not image_data:
                chain = self.prompt | self.llm | StrOutputParser()
                response = chain.stream({
                    "query": query,
                    "sources": sources_formatted,
                    "chat_history": chat_history,
                })
                for chunk in response:
                    yield chunk
            else:
                # Images present: use multimodal format
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    
                messages = []
    
                # ✅ Parse chat history into message objects
                for msg in chat_history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                    else:
                        logger.warning(f"Skipping malformed chat history message: {msg}")
    
                # Build system message with optional grounded context
                system_prompt = self.GROUNDED_PROMPT
                if sources_formatted and sources_formatted != "No relevant sources found.":
                    system_prompt += f"\n\nHere is context information that might be helpful:\n{sources_formatted}"
                messages.append(SystemMessage(content=system_prompt))
    
                # Add human multimodal message
                content_blocks = [{"type": "text", "text": query}]
                for img in image_data:
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mime_type']};base64,{img['base64_data']}"
                        }
                    })
                messages.append(HumanMessage(content=content_blocks))
    
                logger.info(f"Sending {len(content_blocks)-1} image(s) with the prompt.")
    
                # Stream the response from the LLM
                response = self.llm.stream(messages)
                for chunk in response:
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content
    
        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}")
            yield json.dumps({"error": str(e)})


    def get_context_resources_from_db(self, query, user_id, document_id):
        """Get relevant context from the vector store."""
        logger.info(f"Starting chat with query: '{query}'")
        
        # Perform the search
        search_results = self.search(query, user_id, document_id)
        
        # Format the results
        sources_formatted = self.format_sources(search_results)
        
        # Log the amount of context being used
        logger.info(f"Found {len(search_results)} relevant sources")
        logger.debug(f"Total context length: {len(sources_formatted)} characters")
        
        # Add query context to help guide the response
        context_header = (
            f"QUERY: {query}\n"
            f"Number of relevant sources found: {len(search_results)}\n"
            "=================\n\n"
        )
        
        return context_header + sources_formatted 
