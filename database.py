from sqlalchemy import create_engine, Column, String, JSON, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
CA_CERT_PATH = os.path.join(current_dir, 'ca-certificate.crt')

# Database configuration
DB_CONFIG = {
    'username': 'doadmin',
    'password': 'AVNS_vmDiDMks3jF2DaMfYQh',
    'host': 'balela-python-ai-do-user-20395024-0.d.db.ondigitalocean.com',
    'port': '25060',
    'database': 'defaultdb'
}

# Create SQLAlchemy Base
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    # Composite primary key using user_id and document_id
    user_id = Column(String(255), primary_key=True)
    document_id = Column(String(255), primary_key=True)
    
    # Store content and embedding as JSON
    content = Column(JSON)
    embedding = Column(JSON)
    
    # Metadata
    is_book = Column(String(5), default='false')  # Using string to store boolean
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Create an index on user_id and document_id for faster lookups
    __table_args__ = (
        Index('idx_user_doc', user_id, document_id),
    )

def get_database_url():
    # Configure SSL settings for DigitalOcean managed MySQL
    ssl_config = {
        'ssl_ca': CA_CERT_PATH
    }
    
    # Construct the URL with SSL configuration
    url = f"mysql+mysqlconnector://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    
    return url, ssl_config

def init_db():
    try:
        # Get database URL and SSL configuration
        url, ssl_config = get_database_url()
        
        # Create database engine with SSL configuration
        engine = create_engine(url, connect_args=ssl_config)
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        # Create session factory
        Session = sessionmaker(bind=engine)
        
        logger.info("Database initialized successfully")
        return engine, Session
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def save_document_to_db(session, user_id, document_id, content, embedding, is_book=False):
    try:
        # Create new document
        doc = Document(
            user_id=user_id,
            document_id=document_id,
            content=json.dumps(content),
            embedding=json.dumps(embedding),
            is_book=str(is_book).lower(),
            created_at=datetime.utcnow()
        )
        
        # Add and commit
        session.add(doc)
        session.commit()
        logger.info(f"Document saved successfully for user_id: {user_id}, document_id: {document_id}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving document to database: {str(e)}")
        return False

def get_document_from_db(session, user_id, document_id):
    try:
        doc = session.query(Document).filter_by(
            user_id=user_id,
            document_id=document_id
        ).first()
        
        if doc:
            return {
                'content': json.loads(doc.content),
                'embedding': json.loads(doc.embedding),
                'is_book': doc.is_book == 'true',
                'created_at': doc.created_at
            }
        return None
    except Exception as e:
        logger.error(f"Error retrieving document from database: {str(e)}")
        return None 

def delete_document_by_document_id(document_id):
    """Delete a document from the database by its document_id."""
    try:
        # Get database engine and Session
        engine, Session = init_db()
        
        # Create a new session
        session = Session()
        
        try:
            # Delete all documents with matching document_id
            result = session.query(Document).filter_by(document_id=document_id).delete()
            
            # Commit the transaction
            session.commit()
            
            if result > 0:
                logger.info(f"Successfully deleted {result} document(s) with document_id: {document_id}")
                return True
            else:
                logger.warning(f"No documents found with document_id: {document_id}")
                return False
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error during document deletion: {str(e)}")
            return False
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error initializing database connection for deletion: {str(e)}")
        return False
