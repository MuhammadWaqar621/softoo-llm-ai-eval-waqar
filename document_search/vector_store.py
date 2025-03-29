"""
Vector database management for Multilingual RAG System.
"""
import os
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import config
from embedding import EmbeddingManager

class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.
    """
    
    def __init__(self, embedding_manager=None, persist_directory=None):
        """
        Initialize the vector store.
        
        Args:
            embedding_manager: EmbeddingManager instance (will create if None)
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory or config.PERSIST_DIRECTORY
        
        # Initialize embedding manager if not provided
        if embedding_manager is None:
            self.embedding_manager = EmbeddingManager()
        else:
            self.embedding_manager = embedding_manager
        
        self._vector_db = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        print(f"🌐 Vector store initialized with directory: {self.persist_directory}")
    
    def create_vector_database(self, documents: List[Document]) -> None:
        """
        Create and persist a vector database from documents.
        
        Args:
            documents: List of document chunks to add to the vector store
        """
        print("*" * 50)
        print("🌐 Vector Database Creation")
        print("*" * 50)
        
        print(f"🧮 Embedding Model: {self.embedding_manager.model_name}")
        print(f"📊 Creating vector embeddings for {len(documents)} chunks...")
        
        # Get the embedding function
        embedding_function = self.embedding_manager.embedding_model
        
        # Create and persist the vector store
        self._vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=self.persist_directory
        )
        
        # Persist to disk
        print("💾 Persisting vector database...")
        self._vector_db.persist()
        
        print("\n📊 Vector Database Summary:")
        print(f"🔢 Total Embedded Chunks: {len(documents)}")
        print(f"💽 Persist Directory: {self.persist_directory}")
        print("✅ Vector Database Created and Persisted")
        print("*" * 50 + "\n")
    
    def load_vector_database(self) -> bool:
        """
        Load an existing vector database.
        
        Returns:
            True if database loaded successfully, False otherwise
        """
        # Check if database exists
        db_file = os.path.join(self.persist_directory, "chroma.sqlite3")
        
        if not os.path.exists(db_file):
            print(f"❌ No existing vector database found at {db_file}")
            return False
        
        try:
            print(f"📂 Loading existing vector database from {self.persist_directory}...")
            
            # Get the embedding function
            embedding_function = self.embedding_manager.embedding_model
            
            # Load the vector store
            self._vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding_function
            )
            
            # Get collection stats
            collection = self._vector_db._collection
            count = collection.count()
            
            print(f"✅ Successfully loaded vector database with {count} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector database: {str(e)}")
            return False
    
    def get_retriever(self, top_k=None):
        """
        Get a retriever for querying the vector database.
        
        Args:
            top_k: Number of documents to retrieve (default from config)
            
        Returns:
            Retriever object
        """
        if self._vector_db is None:
            if not self.load_vector_database():
                raise ValueError("Vector database not available. Create or load a database first.")
        
        # Use default top_k from config if not specified
        if top_k is None:
            top_k = config.TOP_K_CHUNKS
        
        # Create retriever with search parameters
        retriever = self._vector_db.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        return retriever
    
    def similarity_search(self, query: str, k=None) -> List[Document]:
        """
        Perform similarity search on the vector database.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if self._vector_db is None:
            if not self.load_vector_database():
                raise ValueError("Vector database not available. Create or load a database first.")
        
        # Use default k from config if not specified
        if k is None:
            k = config.TOP_K_CHUNKS
        
        return self._vector_db.similarity_search(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to an existing vector database.
        
        Args:
            documents: List of documents to add
        """
        if self._vector_db is None:
            if not self.load_vector_database():
                # Create new if doesn't exist
                self.create_vector_database(documents)
                return
        
        print(f"📝 Adding {len(documents)} new documents to vector database...")
        
        # Add documents
        self._vector_db.add_documents(documents)
        
        # Persist changes
        self._vector_db.persist()
        
        print("✅ Documents added and persisted to vector database")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self._vector_db is None:
            if not self.load_vector_database():
                return {"error": "Vector database not available"}
        
        try:
            collection = self._vector_db._collection
            count = collection.count()
            
            return {
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_manager.model_name
            }
        except Exception as e:
            return {"error": str(e)}