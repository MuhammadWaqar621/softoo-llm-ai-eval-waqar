"""
Text chunking module for Multilingual RAG System.
"""
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
import config

class DocumentChunker:
    """
    Splits documents into chunks for better embedding and retrieval.
    """
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of each chunk in characters (default from config)
            chunk_overlap: Overlap between chunks in characters (default from config)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        print(f"✂️ Document chunker initialized with size: {self.chunk_size}, overlap: {self.chunk_overlap}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        print("*" * 50)
        print("✂️ Document Chunking Process")
        print("*" * 50)
        
        print(f"📏 Chunk Size: {self.chunk_size}")
        print(f"📏 Chunk Overlap: {self.chunk_overlap}")
        
        # Check for documents in different languages and apply appropriate chunking
        chunked_docs = self._chunk_documents_by_language(documents)
        
        print(f"✅ Created {len(chunked_docs)} document chunks")
        print("\n📊 Chunking Summary:")
        print(f"🔢 Original Documents: {len(documents)}")
        print(f"🔢 Chunked Documents: {len(chunked_docs)}")
        print("*" * 50 + "\n")
        
        return chunked_docs
    
    def _chunk_documents_by_language(self, documents: List[Document]) -> List[Document]:
        """
        Apply appropriate chunking based on document language.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        # Group documents by language if available in metadata
        documents_by_language = {}
        
        for doc in documents:
            # Try to detect language from metadata or content
            lang = self._get_document_language(doc)
            
            if lang not in documents_by_language:
                documents_by_language[lang] = []
            
            documents_by_language[lang].append(doc)
        
        # Print language distribution
        for lang, docs in documents_by_language.items():
            print(f"📚 Found {len(docs)} documents in language: {lang}")
        
        # Initialize the default chunker
        default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # Process documents by language group
        all_chunks = []
        
        for lang, docs in documents_by_language.items():
            # Choose the appropriate chunker for this language
            splitter = self._get_splitter_for_language(lang)
            
            # Split documents
            print(f"🔪 Splitting {len(docs)} {lang} documents...")
            chunks = splitter.split_documents(docs)
            
            # Add language to metadata if not already there
            for chunk in chunks:
                if 'language' not in chunk.metadata:
                    chunk.metadata['language'] = lang
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _get_document_language(self, document: Document) -> str:
        """
        Get or detect the language of a document.
        
        Args:
            document: Document to analyze
            
        Returns:
            Language code or "unknown"
        """
        # Check if language is in metadata
        if 'language' in document.metadata:
            return document.metadata['language']
        
        # Try to detect language from content
        try:
            # If langdetect is available, use it
            from langdetect import detect
            
            # Get a sample of the text (first 1000 chars)
            sample = document.page_content[:1000]
            if sample:
                lang = detect(sample)
                return lang
        except:
            # If detection fails or langdetect is not available
            pass
        
        # Default to unknown
        return "multilingual"  # Default to multilingual handling
    
    def _get_splitter_for_language(self, language: str):
        """
        Get the appropriate text splitter for a language.
        
        Args:
            language: Language code
            
        Returns:
            Text splitter object
        """
        # Special handling for languages that may need different chunking
        if language in ["zh", "ja", "ko", "th"]:
            # For languages without clear word boundaries, character-based is better
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=" "
            )
        elif language in ["ar", "ur", "fa"]:
            # For right-to-left languages
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            # Default recursive character splitter
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )