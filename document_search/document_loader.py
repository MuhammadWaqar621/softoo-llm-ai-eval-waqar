"""
Document loading module for Multilingual RAG System.
"""
import os
import hashlib
import shutil
import chardet
import json
from datetime import datetime
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader, 
    PDFMinerLoader, 
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
import config

class DocumentLoader:
    """
    Document loader that supports multiple file formats and multilingual content.
    """
    
    def __init__(self, docs_dir=None):
        """
        Initialize the document loader.
        
        Args:
            docs_dir: Directory containing documents to process
        """
        self.docs_dir = docs_dir or config.DOCUMENTS_DIRECTORY
        
        # Create necessary directories
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(config.LOG_DIRECTORY, exist_ok=True)
        os.makedirs(config.TEMP_DIRECTORY, exist_ok=True)
        
        # Initialize tracking
        self.successful_files = {ext.strip('.'): [] for ext in config.SUPPORTED_EXTENSIONS}
        self.failed_files = {ext.strip('.'): [] for ext in config.SUPPORTED_EXTENSIONS}
        
        print(f"📂 Document loader initialized with directory: {self.docs_dir}")
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all documents from the configured directory.
        
        Returns:
            List of loaded document objects
        """
        print("*" * 50)
        print("📂 Document Loading Process")
        print("*" * 50)
        
        all_documents = []
        
        # Walk through the directory
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                _, ext = os.path.splitext(file.lower())
                
                # Check if this is a supported file type
                if ext in config.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    
                    # Get the file type without the dot
                    file_type = ext.strip('.')
                    
                    try:
                        # Load document based on type
                        docs = self._load_file(file_path, file_type)
                        
                        if docs:
                            all_documents.extend(docs)
                            self.successful_files[file_type].append(file_path)
                            print(f"✅ Loaded {file_type} file: {file}")
                        else:
                            self.failed_files[file_type].append(file_path)
                            print(f"❌ No documents extracted from: {file}")
                    except Exception as e:
                        self.failed_files[file_type].append(file_path)
                        print(f"❌ Error loading {file_path}: {str(e)}")
        
        # Save loading results
        self._save_loading_log()
        
        # Print summary
        print("\n📊 Document Loading Summary:")
        for ext, files in self.successful_files.items():
            if files:
                print(f"🔢 {ext.upper()} files loaded: {len(files)}")
        
        for ext, files in self.failed_files.items():
            if files:
                print(f"🔢 {ext.upper()} files failed: {len(files)}")
        
        print(f"🔢 Total documents: {len(all_documents)}")
        print("*" * 50 + "\n")
        
        return all_documents
    
    def _load_file(self, file_path: str, file_type: str) -> List[Document]:
        """
        Load a file based on its type.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, txt, etc.)
            
        Returns:
            List of document objects
        """
        if file_type == 'pdf':
            return self._load_pdf(file_path)
        elif file_type == 'txt':
            return self._load_text(file_path)
        elif file_type == 'md':
            return self._load_markdown(file_path)
        elif file_type == 'docx':
            return self._load_docx(file_path)
        else:
            print(f"⚠️ Unsupported file type: {file_type}")
            return []
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file using multiple strategies."""
        # Try PyPDFLoader first
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if docs:
                return docs
        except Exception as e:
            print(f"⚠️ PyPDFLoader failed for {file_path}: {str(e)}")
        
        # If PyPDFLoader fails, try PDFMinerLoader
        try:
            loader = PDFMinerLoader(file_path)
            docs = loader.load()
            if docs:
                return docs
        except Exception as e:
            print(f"⚠️ PDFMinerLoader failed for {file_path}: {str(e)}")
        
        # If direct loading fails, try with a simplified path
        try:
            # Create a temporary copy with a simple filename
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:10]
            temp_file = os.path.join(config.TEMP_DIRECTORY, f"temp_pdf_{file_hash}.pdf")
            
            # Make sure path is normalized
            norm_path = os.path.normpath(os.path.abspath(file_path))
            
            # Copy the file
            shutil.copy2(norm_path, temp_file)
            
            try:
                # Try loading the temporary copy
                loader = PDFMinerLoader(temp_file)
                docs = loader.load()
                
                # Update metadata to point to original file
                for doc in docs:
                    doc.metadata['source'] = file_path
                
                return docs
            finally:
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as e:
            print(f"❌ All loading methods failed for {file_path}: {str(e)}")
        
        return []
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load a text file with encoding detection."""
        try:
            # Detect file encoding
            with open(file_path, 'rb') as rawdata:
                result = chardet.detect(rawdata.read())
            
            # Load with detected encoding
            encoding = result['encoding'] or 'utf-8'
            loader = TextLoader(file_path, encoding=encoding)
            return loader.load()
        except Exception as e:
            print(f"❌ Error loading text file {file_path}: {str(e)}")
            return []
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        """Load a markdown file."""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"❌ Error loading markdown file {file_path}: {str(e)}")
            return []
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load a DOCX file."""
        try:
            loader = Docx2txtLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"❌ Error loading DOCX file {file_path}: {str(e)}")
            return []
    
    def _save_loading_log(self) -> None:
        """Save document loading results to a log file."""
        # Create log data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_data = {
            "timestamp": timestamp,
            "successful": self.successful_files,
            "failed": self.failed_files,
            "summary": {
                "total_successful": sum(len(files) for files in self.successful_files.values()),
                "total_failed": sum(len(files) for files in self.failed_files.values())
            }
        }
        
        # Save detailed log
        log_filename = os.path.join(config.LOG_DIRECTORY, f"document_loading_log_{timestamp}.json")
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # Also save as latest log
        latest_log = os.path.join(config.LOG_DIRECTORY, "latest_loading_log.json")
        with open(latest_log, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Saved loading log to {log_filename}")
        print(f"📝 Updated latest loading log at {latest_log}")