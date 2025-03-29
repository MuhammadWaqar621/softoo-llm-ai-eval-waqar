"""
Main RAG (Retrieval-Augmented Generation) system for multilingual question answering.
"""
import os
import json
from typing import List, Any
import time

from groq import Groq
from langchain.docstore.document import Document
import config
from document_loader import DocumentLoader
from chunking import DocumentChunker
from embedding import EmbeddingManager
from vector_store import VectorStore

class RAGSystem:
    """
    Retrieval-Augmented Generation system for multilingual document question-answering.
    """
    
    def __init__(self):
        """Initialize the RAG system components."""
        print("*" * 50)
        print("🚀 Initializing Multilingual RAG System")
        print("*" * 50)
        
        # Initialize embedding manager
        print("📊 Setting up embedding model...")
        self.embedding_manager = EmbeddingManager()
        
        # Initialize vector store
        print("🗄️ Setting up vector database...")
        self.vector_store = VectorStore(self.embedding_manager)
        
        # Initialize Groq client for LLM
        print("🤖 Setting up LLM client...")
        self.groq_client = Groq(api_key=config.APIKEY)
        
        # Create directories if they don't exist
        for directory in [config.DOCUMENTS_DIRECTORY, config.PERSIST_DIRECTORY, config.LOG_DIRECTORY]:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Initialization Complete\n")
    
    def ingest_documents(self) -> None:
        """
        Ingest documents from the configured directory.
        
        Workflow:
        1. Load documents
        2. Chunk documents
        3. Create vector database
        """
        # Create document loader
        document_loader = DocumentLoader()
        
        # Load documents
        documents = document_loader.load_all_documents()
        
        if not documents:
            print("❌ No documents were loaded. Check your data directory.")
            return
        
        # Create document chunker
        chunker = DocumentChunker()
        
        # Chunk documents
        chunked_docs = chunker.chunk_documents(documents)
        
        # Create vector database
        self.vector_store.create_vector_database(chunked_docs)
        
        print("✅ Document ingestion completed successfully")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve most relevant document chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve (default from config)
            
        Returns:
            List of relevant document chunks
        """
        print("*" * 50)
        print("🔍 Retrieving Relevant Chunks")
        print("*" * 50)
        
        # Use default from config if not specified
        if top_k is None:
            top_k = config.TOP_K_CHUNKS
        
        # Perform similarity search
        try:
            relevant_chunks = self.vector_store.similarity_search(query, k=top_k)
        except ValueError as e:
            print(f"❌ Error: {str(e)}")
            return []
        
        print("\n📊 Retrieval Summary:")
        print(f"❓ Query: {query}")
        print(f"🔢 Retrieved Chunks: {len(relevant_chunks)}")
        
        # Display summary of retrieved chunks
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.metadata.get('source', 'Unknown source')
            content_preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
            print(f"📄 Chunk {i} from {source}: {content_preview}")
        
        print("*" * 50 + "\n")
        
        return relevant_chunks
    
    def generate_answer(self, query: str, context: List[Document]) -> str:
        """
        Generate an answer using Groq API based on retrieved context.
        
        Args:
            query: User's question
            context: List of relevant document chunks
            
        Returns:
            Generated answer
        """
        # Combine context chunks
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        # Get sources for citation
        sources = []
        for doc in context:
            if 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
        
        # Construct prompt
        prompt = f"""Context information from documents:
{context_text}

Question: {query}

Provide a detailed and accurate answer based on the given context information.
Your answer should be comprehensive and directly address the question.
If the context doesn't contain relevant information to answer the question, 
say "I don't have enough information to answer this question."
"""
        
        try:
            # Generate answer using Groq
            response = self.groq_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful multilingual document Q&A assistant. Answer questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Add sources information
            if sources:
                sources_text = "\n\nSources:\n" + "\n".join([f"- {s}" for s in sources])
                answer += sources_text
            
            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return f"Sorry, I couldn't generate an answer due to an error: {str(e)}"
    
    def answer_question(self, query: str) -> str:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query: User's question
            
        Returns:
            Generated answer
        """
        start_time = time.time()
        
        # Retrieve relevant document chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Generate answer
        answer = self.generate_answer(query, relevant_chunks)
        
        # Calculate response time
        elapsed_time = time.time() - start_time
        timing_info = f"\n\n[Response generated in {elapsed_time:.2f} seconds]"
        
        return answer + timing_info
    
    def interactive_qa(self) -> None:
        """
        Interactive Question & Answer session.
        
        Allows users to ask questions and get answers from the RAG system.
        """
        print("\n===== Multilingual RAG Q&A System =====")
        print("Ask questions about your documents. Type 'exit' to quit.")
        print("Special commands:")
        print("  - 'status': Check system status")
        print("  - 'loaded': List successfully loaded documents")
        print("  - 'exit': Quit the application")
        
        while True:
            # Get user query
            query = input("\nYour question: ").strip()
            
            # Exit condition
            if query.lower() == 'exit':
                break
                
            # Check system status
            elif query.lower() == 'status':
                self._show_system_status()
                continue
                
            # List loaded documents
            elif query.lower() == 'loaded':
                self._show_loaded_documents()
                continue
            
            # Process regular questions
            try:
                # Get answer
                answer = self.answer_question(query)
                
                print("\n=== Answer ===")
                print(answer)
            
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    
    def _show_system_status(self) -> None:
        """Display the current system status."""
        print("\n=== System Status ===")
        
        # Vector DB status
        db_stats = self.vector_store.get_collection_stats()
        if "error" in db_stats:
            print(f"📊 Vector Database: Not available ({db_stats['error']})")
        else:
            print(f"📊 Vector Database: Available")
            print(f"  - Document Count: {db_stats.get('document_count', 'Unknown')}")
            print(f"  - Embedding Model: {db_stats.get('embedding_model', 'Unknown')}")
        
        # Embedding model status
        print(f"\n📊 Embedding Model: {self.embedding_manager.model_name}")
        try:
            dim = self.embedding_manager.get_embedding_dimension()
            print(f"  - Embedding Dimension: {dim}")
        except:
            print("  - Embedding Model: Not loaded")
        
        # LLM status
        print(f"\n📊 LLM Configuration:")
        print(f"  - Provider: {config.LLM_PROVIDER}")
        print(f"  - Model: {config.LLM_MODEL}")
        print(f"  - Temperature: {config.LLM_TEMPERATURE}")
    
    def _show_loaded_documents(self) -> None:
        """Display the list of successfully loaded documents."""
        try:
            log_file = os.path.join(config.LOG_DIRECTORY, "latest_loading_log.json")
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                print("\n=== Successfully Loaded Documents ===")
                
                for ext, files in log_data['successful'].items():
                    if files:
                        print(f"\n{ext.upper()} Files ({len(files)}):")
                        for i, file_path in enumerate(files, 1):
                            print(f"  {i}. {file_path}")
                
                print(f"\nTotal Documents: {log_data['summary']['total_successful']}")
            else:
                print("No loading log available. Run document ingestion first.")
        except Exception as e:
            print(f"Error reading loaded documents: {str(e)}")


def main():
    """
    Main entry point for the RAG system.
    
    Handles initial setup and launches interactive Q&A.
    """
    # Create RAG system
    rag_system = RAGSystem()
    
    # Check if vector database exists
    vector_db_exists = os.path.exists(os.path.join(config.PERSIST_DIRECTORY, "chroma.sqlite3"))
    
    if not vector_db_exists:
        print("🛠️ No existing vector database found.")
        print("🚧 Creating vector database by ingesting documents...")
        rag_system.ingest_documents()
    
    # Start interactive Q&A
    rag_system.interactive_qa()


if __name__ == "__main__":
    main()