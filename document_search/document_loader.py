import logging
from pathlib import Path
import PyPDF2
from typing import List, Dict
from langdetect import detect

from config import Config

class DocumentLoader:
    """
    A class to load and extract text from various document types.
    
    Supports PDF and text file formats with multi-language detection.
    """
    
    def __init__(self, 
                data: Path = Config.DATA_DIR):
        """
        Initialize DocumentLoader with specified directories.
        
        Args:
            pdf_dir (Path): Directory containing PDF files
            text_dir (Path): Directory containing text files
        """
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=Config.LOG_LEVEL)
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (Path): Path to the PDF file
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join(page.extract_text() for page in reader.pages)
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _extract_text_from_txt(self, txt_path: Path) -> str:
        """
        Extract text from a text file.
        
        Args:
            txt_path (Path): Path to the text file
        
        Returns:
            str: Content of the text file
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error reading {txt_path}: {e}")
            return ""
    
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all documents from PDF and text directories.
        
        Returns:
            List[Dict[str, str]]: List of document dictionaries
        """
        documents = []
        all_files = [file for file in self.data.rglob('*') if file.is_file()]   
        for file in all_files:
            
             # Process PDF files
            if file.suffix.lower() == '.pdf':
                text = self._extract_text_from_pdf(file)
                documents.append(self._process_document(file, text))
        
            # Process Text files
            if file.suffix.lower() == '.txt':
                text = self._extract_text_from_txt(file)
                documents.append(self._process_document(file, text))
        
        self.logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _process_document(self, file_path: Path, text: str) -> Dict[str, str]:
        """
        Process a document and extract metadata.
        
        Args:
            file_path (Path): Path to the document
            text (str): Extracted text content
        
        Returns:
            Dict[str, str]: Document metadata and content
        """
        try:
            language = detect(text) if text else 'unknown'
        except Exception:
            language = 'unknown'
        
        return {
            'filename': file_path.name,
            'text': text,
            'language': language,
            'path': str(file_path)
        }