import os
import warnings
import chardet
from typing import List, Dict, Any, Tuple, Set
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Data processing and embedding libraries
import torch
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

# Groq for language model
from groq import Groq

# Import API key from config
from config import APIKEY

# Configuration Constants
DOCUMENTS_DIRECTORY = "data"  # Root directory containing documents
PERSIST_DIRECTORY = "vector_db"  # Directory to store vector database
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
LOG_DIRECTORY = "logs"  # Directory to store loading logs

class DocumentQASystem:
    def __init__(self):
        """
        Initialize the Document QA System
        - Set up embedding model
        - Initialize Groq client
        - Create persist directory
        - Create logs directory
        """
        print("*" * 50)
        print("🚀 Initializing Document QA System")
        print("*" * 50)
        
        # Initialize embedding model for converting text to vectors
        print(f"📊 Loading Embedding Model: {EMBEDDING_MODEL}")
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize Groq client for generating answers
        print("🤖 Initializing Groq Client")
        self.groq_client = Groq(api_key=APIKEY)
        
        # Ensure vector database directory exists
        print(f"📁 Ensuring Persist Directory: {PERSIST_DIRECTORY}")
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Ensure logs directory exists
        print(f"📁 Ensuring Logs Directory: {LOG_DIRECTORY}")
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        
        print("✅ Initialization Complete\n")

    def load_documents(self) -> Tuple[List[Document], Dict[str, List[str]]]:
        """
        Load documents from the specified directory
        
        Workflow:
        1. Load PDF documents using multiple loaders
        2. Load text documents
        3. Track successful and failed document loads
        4. Combine and return all documents and tracking info
        
        Returns:
        - List of loaded documents
        - Dictionary with successful and failed file lists
        """
        print("*" * 50)
        print("📂 Document Loading Process")
        print("*" * 50)

        # Tracking variables for logging
        total_pdf_files = 0
        total_text_files = 0
        all_documents = []
        
        # Tracking loaded and failed files
        loaded_files = {
            "successful": {
                "pdf": [],
                "txt": []
            },
            "failed": {
                "pdf": [],
                "txt": []
            }
        }
        
        # Get all PDF files in the directory
        all_pdf_files = set()
        for root, _, files in os.walk(DOCUMENTS_DIRECTORY):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    all_pdf_files.add(file_path)
        
        print(f"🔢 Found {len(all_pdf_files)} total PDF files")

        # Load PDF documents using PyPDFLoader
        try:
            print("🔍 Loading PDF files with PyPDFLoader...")
            pypdf_loader = DirectoryLoader(
                DOCUMENTS_DIRECTORY, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            pdf_docs_pypdf = pypdf_loader.load()
            
            # Track successfully loaded PDFs with PyPDFLoader
            pypdf_loaded_files = set()
            for doc in pdf_docs_pypdf:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    pypdf_loaded_files.add(doc.metadata['source'])
                    loaded_files["successful"]["pdf"].append(doc.metadata['source'])
            
            total_pdf_files += len(pypdf_loaded_files)
            all_documents.extend(pdf_docs_pypdf)
            print(f"✅ Loaded {len(pypdf_loaded_files)} PDFs with PyPDFLoader")
            
            # Track failed PDFs with PyPDFLoader
            pypdf_failed = all_pdf_files - pypdf_loaded_files
            
        except Exception as e:
            print(f"❌ Error loading PDFs with PyPDFLoader: {e}")
            pypdf_failed = all_pdf_files
            pypdf_loaded_files = set()

        # Load PDF documents using PDFMinerLoader as a backup for failed files
        pdfminer_loaded_files = set()
        if pypdf_failed:
            try:
                print(f"🔍 Attempting to load {len(pypdf_failed)} failed PDFs with PDFMinerLoader...")
                
                for file_path in pypdf_failed:
                    try:
                        # Try to handle long/complex file paths by normalizing
                        norm_path = os.path.normpath(os.path.abspath(file_path))
                        try:
                            # First attempt - direct loading
                            loader = PDFMinerLoader(norm_path)
                            docs = loader.load()
                            all_documents.extend(docs)
                            pdfminer_loaded_files.add(file_path)
                            loaded_files["successful"]["pdf"].append(file_path)
                        except Exception as inner_e:
                            print(f"⚠️ First attempt failed for {file_path}, trying alternative method...")
                            # Second attempt - use a temporary copy with simplified name
                            temp_dir = os.path.join(os.getcwd(), "temp_pdf_processing")
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            # Create a simple filename based on hash of original path
                            import hashlib
                            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:10]
                            temp_file = os.path.join(temp_dir, f"temp_pdf_{file_hash}.pdf")
                            
                            # Copy the file with simple name
                            import shutil
                            shutil.copy2(norm_path, temp_file)
                            
                            try:
                                # Try loading with the simple filename
                                loader = PDFMinerLoader(temp_file)
                                docs = loader.load()
                                
                                # Restore original source in metadata
                                for doc in docs:
                                    doc.metadata['source'] = file_path
                                
                                all_documents.extend(docs)
                                pdfminer_loaded_files.add(file_path)
                                loaded_files["successful"]["pdf"].append(file_path)
                                print(f"✅ Successfully loaded using temporary copy: {file_path}")
                            except Exception as temp_e:
                                print(f"❌ Both loading methods failed for {file_path}: {temp_e}")
                                loaded_files["failed"]["pdf"].append(file_path)
                            finally:
                                # Clean up temp file
                                if os.path.exists(temp_file):
                                    try:
                                        os.remove(temp_file)
                                    except:
                                        pass
                    except Exception as e:
                        print(f"❌ Failed to load {file_path} with PDFMinerLoader: {e}")
                        loaded_files["failed"]["pdf"].append(file_path)
                
                total_pdf_files += len(pdfminer_loaded_files)
                print(f"✅ Loaded {len(pdfminer_loaded_files)} PDFs with PDFMinerLoader")
            except Exception as e:
                print(f"❌ Error in PDFMinerLoader process: {e}")
        
        # Update list of failed PDFs
        all_failed_pdfs = all_pdf_files - pypdf_loaded_files - pdfminer_loaded_files
        loaded_files["failed"]["pdf"].extend(list(all_failed_pdfs))

        # Load text documents
        print("📄 Loading Text files...")
        all_text_files = set()
        for root, _, files in os.walk(DOCUMENTS_DIRECTORY):
            for file in files:
                if file.lower().endswith('.txt'):
                    file_path = os.path.join(root, file)
                    all_text_files.add(file_path)
                    try:
                        # Detect file encoding
                        with open(file_path, 'rb') as rawdata:
                            result = chardet.detect(rawdata.read())
                        
                        # Load text document with detected encoding
                        loader = TextLoader(file_path, encoding=result['encoding'] or 'utf-8')
                        text_docs = loader.load()
                        total_text_files += len(text_docs)
                        all_documents.extend(text_docs)
                        print(f"✅ Loaded text file: {file}")
                        loaded_files["successful"]["txt"].append(file_path)
                    except Exception as e:
                        print(f"❌ Error loading text file {file_path}: {e}")
                        loaded_files["failed"]["txt"].append(file_path)

        # Print detailed loading summary
        print("\n📊 Document Loading Summary:")
        print(f"🔢 Total PDF files found: {len(all_pdf_files)}")
        print(f"🔢 Total PDF files loaded: {total_pdf_files}")
        print(f"🔢 Total PDF files failed: {len(all_pdf_files) - total_pdf_files}")
        print(f"🔢 Total Text files loaded: {total_text_files}")
        print(f"🔢 Total Text files failed: {len(loaded_files['failed']['txt'])}")
        print(f"🔢 Total documents loaded: {len(all_documents)}")
        print("*" * 50 + "\n")

        # Save loading results to log file
        self._save_loading_log(loaded_files)
        
        return all_documents, loaded_files

    def _save_loading_log(self, loaded_files: Dict):
        """Save document loading results to a timestamped log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIRECTORY, f"document_loading_log_{timestamp}.json")
        
        # Add summary statistics
        log_data = {
            "timestamp": timestamp,
            "summary": {
                "total_pdf_found": len(loaded_files["successful"]["pdf"]) + len(loaded_files["failed"]["pdf"]),
                "total_pdf_loaded": len(loaded_files["successful"]["pdf"]),
                "total_pdf_failed": len(loaded_files["failed"]["pdf"]),
                "total_txt_loaded": len(loaded_files["successful"]["txt"]),
                "total_txt_failed": len(loaded_files["failed"]["txt"])
            },
            "loaded_files": loaded_files
        }
        
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Saved loading log to {log_filename}")
        
        # Also create latest.json for quick reference
        latest_log = os.path.join(LOG_DIRECTORY, "latest_loading_log.json")
        with open(latest_log, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Updated latest loading log at {latest_log}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better embedding and retrieval
        
        Workflow:
        1. Use RecursiveCharacterTextSplitter to create document chunks
        2. Maintains context with overlap between chunks
        """
        print("*" * 50)
        print("✂️ Document Chunking Process")
        print("*" * 50)
        
        print(f"📏 Chunk Size: {CHUNK_SIZE}")
        print(f"📏 Chunk Overlap: {CHUNK_OVERLAP}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
        print("🔪 Splitting documents into chunks...")
        chunked_docs = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunked_docs)} document chunks")
        print("\n📊 Chunking Summary:")
        print(f"🔢 Original Documents: {len(documents)}")
        print(f"🔢 Chunked Documents: {len(chunked_docs)}")
        print("*" * 50 + "\n")
        
        return chunked_docs

    def create_vector_database(self, chunked_docs: List[Document]):
        """
        Create and persist vector database
        
        Workflow:
        1. Convert document chunks to vector embeddings
        2. Store vectors in Chroma vector database
        3. Persist the database for later use
        """
        print("*" * 50)
        print("🌐 Vector Database Creation")
        print("*" * 50)
        
        print(f"🧮 Embedding Model: {EMBEDDING_MODEL}")
        print("📊 Creating vector embeddings...")
        
        vector_db = Chroma.from_documents(
            chunked_docs, 
            self.embeddings, 
            persist_directory=PERSIST_DIRECTORY
        )
        
        print("💾 Persisting vector database...")
        vector_db.persist()
        
        print("\n📊 Vector Database Summary:")
        print(f"🔢 Total Embedded Chunks: {len(chunked_docs)}")
        print(f"💽 Persist Directory: {PERSIST_DIRECTORY}")
        print("✅ Vector Database Created and Persisted")
        print("*" * 50 + "\n")

    def ingest_documents(self):
        """
        Main document ingestion pipeline
        
        Workflow:
        1. Load documents from directory
        2. Chunk documents
        3. Create vector database
        """
        # Load documents
        documents, loaded_files = self.load_documents()
        
        # Chunk documents
        chunked_docs = self.chunk_documents(documents)
        
        # Create vector database
        self.create_vector_database(chunked_docs)
        
        return loaded_files

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3):
        """
        Retrieve most relevant document chunks for a query
        
        Workflow:
        1. Load existing vector database
        2. Use similarity search to find relevant chunks
        """
        print("*" * 50)
        print("🔍 Retrieving Relevant Chunks")
        print("*" * 50)
        
        print("💾 Loading vector database...")
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=self.embeddings
        )
        
        print(f"🔬 Performing similarity search (top {top_k} chunks)")
        retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
        relevant_chunks = retriever.get_relevant_documents(query)
        
        print("\n📊 Retrieval Summary:")
        print(f"❓ Query: {query}")
        print(f"🔢 Retrieved Chunks: {len(relevant_chunks)}")
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk.metadata.get('source', 'Unknown source')
            print(f"📄 Chunk {i} from {source}: {chunk.page_content[:100]}...")
        print("*" * 50 + "\n")
        
        return relevant_chunks

    def generate_answer(self, query: str, context: List[Document]) -> str:
        """
        Generate an answer using Groq API based on retrieved context
        
        Workflow:
        1. Format context documents
        2. Create a prompt with context and query
        3. Call Groq API to generate answer
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
        prompt = f"""Context from documents:
{context_text}

Question: {query}

Provide a detailed and accurate answer based on the given context. 
Include information about which documents were used to generate this answer."""

        try:
            # Generate answer using Groq
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful document Q&A assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Add sources information
            answer = response.choices[0].message.content
            sources_text = "\n\nSources:\n" + "\n".join([f"- {s}" for s in sources])
            
            return answer + sources_text
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Could not generate an answer."

    def interactive_qa(self):
        """
        Interactive Question & Answer loop
        
        Workflow:
        1. Accept user queries
        2. Retrieve relevant document chunks
        3. Generate and display answers
        """
        print("\n===== Document QA System =====")
        print("Ask questions about your documents. Type 'exit' to quit.")
        print("Special commands:")
        print("  - 'status': Check document loading status")
        print("  - 'loaded': List successfully loaded documents")
        print("  - 'failed': List failed documents")
        print("  - 'exit': Quit the application")
        
        while True:
            # Get user query
            query = input("\nYour question: ").strip()
            
            # Exit condition
            if query.lower() == 'exit':
                break
                
            # Check loading status
            elif query.lower() == 'status':
                self._show_loading_status()
                continue
                
            # List loaded documents
            elif query.lower() == 'loaded':
                self._show_loaded_documents()
                continue
                
            # List failed documents
            elif query.lower() == 'failed':
                self._show_failed_documents()
                continue
            
            try:
                # Retrieve relevant chunks
                relevant_chunks = self.retrieve_relevant_chunks(query)
                
                # Generate answer
                if relevant_chunks:
                    answer = self.generate_answer(query, relevant_chunks)
                    print("\n=== Answer ===")
                    print(answer)
                else:
                    print("No relevant information found.")
            
            except Exception as e:
                print(f"An error occurred: {e}")

    def _show_loading_status(self):
        """Display the current document loading status"""
        try:
            log_file = os.path.join(LOG_DIRECTORY, "latest_loading_log.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                print("\n=== Document Loading Status ===")
                print(f"Last updated: {log_data['timestamp']}")
                print("\nSummary:")
                summary = log_data['summary']
                print(f"  PDF files: {summary['total_pdf_loaded']}/{summary['total_pdf_found']} loaded")
                print(f"  Text files: {summary['total_txt_loaded']}/{summary['total_txt_loaded'] + summary['total_txt_failed']} loaded")
                print(f"  Total successful: {summary['total_pdf_loaded'] + summary['total_txt_loaded']}")
                print(f"  Total failed: {summary['total_pdf_failed'] + summary['total_txt_failed']}")
            else:
                print("No loading log available. Run document ingestion first.")
        except Exception as e:
            print(f"Error reading loading status: {e}")

    def _show_loaded_documents(self):
        """Display the list of successfully loaded documents"""
        try:
            log_file = os.path.join(LOG_DIRECTORY, "latest_loading_log.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                print("\n=== Successfully Loaded Documents ===")
                
                # Show PDFs
                pdf_files = log_data['loaded_files']['successful']['pdf']
                print(f"\nPDF Files ({len(pdf_files)}):")
                for i, file_path in enumerate(pdf_files, 1):
                    print(f"  {i}. {file_path}")
                
                # Show TXT files
                txt_files = log_data['loaded_files']['successful']['txt']
                print(f"\nText Files ({len(txt_files)}):")
                for i, file_path in enumerate(txt_files, 1):
                    print(f"  {i}. {file_path}")
            else:
                print("No loading log available. Run document ingestion first.")
        except Exception as e:
            print(f"Error reading loaded documents: {e}")

    def _show_failed_documents(self):
        """Display the list of documents that failed to load"""
        try:
            log_file = os.path.join(LOG_DIRECTORY, "latest_loading_log.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                print("\n=== Failed Documents ===")
                
                # Show PDFs
                pdf_files = log_data['loaded_files']['failed']['pdf']
                print(f"\nPDF Files ({len(pdf_files)}):")
                for i, file_path in enumerate(pdf_files, 1):
                    print(f"  {i}. {file_path}")
                
                # Show TXT files
                txt_files = log_data['loaded_files']['failed']['txt']
                print(f"\nText Files ({len(txt_files)}):")
                for i, file_path in enumerate(txt_files, 1):
                    print(f"  {i}. {file_path}")
            else:
                print("No loading log available. Run document ingestion first.")
        except Exception as e:
            print(f"Error reading failed documents: {e}")

def main():
    """
    Main execution flow
    
    Workflow:
    1. Initialize QA System
    2. Ingest documents if no existing database
    3. Start interactive Q&A
    """
    # Create QA System instance
    qa_system = DocumentQASystem()
    
    # Check if vector database exists
    if not os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
        print("🛠️ No existing vector database found.")
        print("🚧 Creating vector database...\n")
        qa_system.ingest_documents()
    
    # Start interactive Q&A
    qa_system.interactive_qa()

if __name__ == "__main__":
    main()