import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.client import WeaviateClient
from weaviate.collections import Collection
import weaviate.classes.query as wq
from weaviate.classes.config import Configure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateVectorStore:
    def __init__(self):
        # Get Weaviate configuration from environment variables
        weaviate_url = os.getenv("WEAVIATE_URL", "http://209.97.187.195:8087")
        
        # Parse URL for proper connection
        parsed_url = urlparse(weaviate_url)
        host = parsed_url.hostname
        port = parsed_url.port or 8087
        protocol = parsed_url.scheme
        
        try:
        # Initialize Weaviate client with v4 syntax
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=8080,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=False,
                additional_config=AdditionalConfig(
                    timeout=Timeout(timeout_config=120)
                )
            )
            
            logger.info(f"Successfully connected to Weaviate at {weaviate_url}")
        
            # Define the class name for documents
            self.class_name = "Document"
            
                # Create schema if it doesn't exist
            self._create_schema()
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {str(e)}")
            raise

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        return (np.array(vector) / norm).tolist() if norm != 0 else vector
    
    def _create_schema(self):
        """Create the Weaviate schema if it doesn't exist."""
        try:
            # Check if collection exists
            try:
                collection = self.client.collections.get(self.class_name)
                logger.info(f"Collection {self.class_name} already exists")
                return
            except Exception:
                # Collection doesn't exist, create it
                pass

            # Create new collection with HNSW and cosine distance
            collection = self.client.collections.create(
                name=self.class_name,
                vectorizer_config=None,  # No vectorizer since we'll provide vectors
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=Configure.VectorDistances.COSINE
                ),
                properties=[
                        {
                            "name": "content",
                            "dataType": ["text"],
                        "description": "The content of the document",
                        "indexInverted": True,  # Enable BM25 text search
                        },
                        {
                            "name": "user_id",
                        "dataType": ["text"],
                        "description": "ID of the user who owns this document"
                        },
                        {
                            "name": "document_id",
                        "dataType": ["text"],
                        "description": "Unique identifier for the document"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                        "description": "Additional metadata"
                    },
                    {
                        "name": "is_book",
                        "dataType": ["boolean"],
                        "description": "Whether the document is a book"
                    }
                ]
            )
            logger.info(f"Created collection {self.class_name} with cosine distance")
            
        except Exception as e:
            if "already exists" not in str(e):
                logger.error(f"Error creating schema: {str(e)}")
            raise

    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     user_id: str, document_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Add documents with their embeddings to Weaviate."""
        try:
            # Get the collection
            collection = self.client.collections.get(self.class_name)
            
            # Normalize embeddings
            normalized_embeddings = [self._normalize_vector(e) for e in embeddings]
            
            # Create a batch object
            with collection.batch.fixed_size(batch_size=100) as batch:
                # Add each document to the batch
                for text, embedding in zip(texts, normalized_embeddings):
                    # Extract is_book flag from metadata if available
                    is_book = metadata.get("is_book", False) if metadata else False
                    
                    properties = {
                        "content": text,
                        "user_id": user_id,
                        "document_id": document_id,
                        "metadata": str(metadata or {}),
                        "is_book": is_book
                    }
                    
                    # Add object with normalized vector
                    batch.add_object(
                        properties=properties,
                        vector=embedding
                    )
                    
            logger.info(f"Successfully added documents for user_id: {user_id}, document_id: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to Weaviate: {str(e)}")
            return False

    def search(self, query, user_id, document_id=None):
        """Enhanced search using hybrid approach with proper vector normalization."""
        logger.info(f"Searching for query: '{query}' for user_id: '{user_id}' and document_id: '{document_id}'")
        try:
            # Get and normalize query embedding
            if hasattr(self, 'embeddings'):
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = self._normalize_vector(query_embedding)
            else:
                logger.error("No embeddings model available")
                return []
            
            # Get the collection
            collection = self.client.collections.get(self.class_name)
            
            # Build the filter
            if document_id:
                base_filter = wq.Filter.by_property("document_id").equal(document_id)
                chunk_filter = wq.Filter.by_property("document_id").like(f"{document_id}_chunk_*")
                doc_filter = base_filter | chunk_filter
                combined_filter = wq.Filter.by_property("user_id").equal(user_id) & doc_filter
            else:
                combined_filter = wq.Filter.by_property("user_id").equal(user_id)

            # Try hybrid search with proper configuration
            try:
                response = collection.query.hybrid(
                    query=query,
                    vector=query_embedding,
                    limit=20,
                    alpha=0.7,  # Weight between vector (0.7) and keyword (0.3) search
                    filters=combined_filter,
                    return_metadata=wq.MetadataQuery(
                        distance=True,
                        score=True
                    ),
                    return_properties=["content", "document_id", "metadata", "is_book"]
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
                # Fallback to pure vector search
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=20,
                    return_metadata=wq.MetadataQuery(distance=True),
                    filters=combined_filter,
                    return_properties=["content", "document_id", "metadata", "is_book"]
                )
            
            if not response.objects:
                logger.warning("No results found in search")
                return []
            
            # Process results with improved scoring
            processed_docs = {}
            for obj in response.objects:
                base_doc_id = obj.properties["document_id"].split("_chunk_")[0]
                
                # Calculate hybrid score
                vector_score = 1 - float(obj.metadata.distance or 0)
                bm25_score = float(obj.metadata.score or 0) / 3.0  # Normalize BM25 score
                
                # Combine scores with weights matching alpha
                score = (0.7 * vector_score) + (0.3 * bm25_score)
                
                # Keyword boost
                relevance_boost = 0
                keywords = query.lower().split()
                content_lower = obj.properties["content"].lower()
                for keyword in keywords:
                    if keyword in content_lower:
                        relevance_boost += 0.1
                
                final_score = min(1.0, score + relevance_boost)
                
                # Only consider if score meets threshold
                if final_score >= 0.65:  # Adjust threshold as needed
                    if base_doc_id not in processed_docs or final_score > processed_docs[base_doc_id]["score"]:
                        try:
                            metadata = obj.properties.get("metadata", {})
                            if isinstance(metadata, str):
                                metadata = eval(metadata)
                        except:
                            metadata = {}
                        
                        processed_docs[base_doc_id] = {
                            "content": obj.properties["content"],
                            "score": final_score,
                            "metadata": metadata,
                            "is_book": obj.properties.get("is_book", False)
                        }
            
            # Format results
            results = []
            for doc_id, doc_info in processed_docs.items():
                has_math = any(char in doc_info["content"] for char in "+-*/^=()[]{}\\")
                content = doc_info["content"]
                
                if has_math:
                    content = content.replace("\\", "\\\\")
                
                results.append({
                    "metadata": {
                        "content": content,
                        "has_math": has_math,
                        "document_id": doc_id,
                        "is_book": doc_info["is_book"],
                        **doc_info["metadata"]
                    },
                    "score": doc_info["score"]
                })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            if results:
                logger.info(f"Found {len(results)} relevant results above threshold")
                logger.debug(f"Top result score: {results[0]['score']:.3f}")
                logger.debug(f"Query: '{query}'")
            
            return results[:10]
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def filtered_similarity_search(self, query_embedding: List[float], user_id: str, 
                                document_id: str, limit: int = 5) -> List[Document]:
        """Search with filters for both user_id and document_id."""
        try:
            # Get the collection
            collection = self.client.collections.get(self.class_name)
            
            # Execute the query with filters for both user_id and document_id
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=wq.MetadataQuery(distance=True),
                filters=wq.Filter.by_property("user_id").equal(user_id) & 
                        wq.Filter.by_property("document_id").equal(document_id)
            )
            
            # Process results
            documents = []
            for obj in response.objects:
                # Same processing as in similarity_search
                metadata = {
                    "document_id": obj.properties["document_id"],
                    "distance": obj.metadata.distance,
                }
                
                if "is_book" in obj.properties:
                    metadata["is_book"] = obj.properties["is_book"]
                
                if obj.properties.get("metadata"):
                    try:
                        additional_metadata = eval(obj.properties["metadata"])
                        if isinstance(additional_metadata, dict):
                            metadata.update(additional_metadata)
                    except:
                        pass
                
                    doc = Document(
                    page_content=obj.properties["content"],
                    metadata=metadata
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error performing filtered similarity search: {str(e)}")
            return []

    def delete_documents(self, document_id: str) -> bool:
        """Delete documents by document_id from Weaviate."""
        try:
            # Get the collection
            collection = self.client.collections.get(self.class_name)
            
            # Create filters for both main document and any chunks
            base_filter = wq.Filter.by_property("document_id").equal(document_id)
            chunk_filter = wq.Filter.by_property("document_id").like(f"{document_id}_chunk_*")
            
            try:
                # Delete main document and any chunks
                delete_result = collection.data.delete_many(
                    where=base_filter | chunk_filter
                )
                
                # In Weaviate v4, delete_many returns a DeleteManyReturn object
                # We just need to check if the operation was successful
                if delete_result is not None:
                    logger.info(f"Successfully deleted documents from Weaviate with document_id: {document_id}")
                    return True
                else:
                    logger.warning(f"No documents found in Weaviate with document_id: {document_id}")
                    return False
                
            except Exception as e:
                logger.error(f"Error during Weaviate deletion: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}")
            return False 

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if hasattr(self, 'client'):
            self.client.close() 
