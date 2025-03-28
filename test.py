import os
import glob
from typing import List

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from pathlib import Path

class RAGChatbot:
    def __init__(
        self, 
        data_directory: Path = 'document_search/data', 
        embedding_model: str = 'nomic-embed-text',
        groq_model: str = 'mixtral-8x7b-32768'
    ):
        """
        Initialize RAG Chatbot with document loading and embedding setup
        
        Args:
            data_directory (str): Path to directory containing documents
            embedding_model (str): Embedding model name
            groq_model (str): Groq LLM model name
        """
        self.data_directory = data_directory
        self.embedding_model = embedding_model
        self.groq_model = groq_model
        
        # Setup embeddings
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Vector store
        self.vectorstore = None
        
        # LLM 
        self.llm = ChatGroq(
            temperature=0.3, 
            model_name=self.groq_model,
            api_key="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"
        )
        
    def load_documents(self) -> List[Document]:
        """
        Recursively load PDF and text files from data directory
        
        Returns:
            List of loaded documents
        """
        all_documents = []
        
        # PDF loader
        all_files = [file for file in self.data_directory.rglob('*') if file.is_file()]   
        print(all_files)

        pdf_files = [file for file in all_files if file.suffix.lower() == ".pdf"]
        pdf_docs = [
            doc for loader in [PyPDFLoader(path) for path in pdf_files]
            for doc in loader.load()
        ]
        all_documents.extend(pdf_docs)
        
        # Text file loader
        txt_paths = glob.glob(os.path.join(self.data_directory, '**', '*.txt'), recursive=True)
        txt_docs = [
            doc for loader in [TextLoader(path, encoding='utf-8') for path in txt_paths]
            for doc in loader.load()
        ]
        all_documents.extend(txt_docs)
        
        return all_documents
    
    def prepare_vector_store(self):
        """
        Load documents, split into chunks, and create vector store
        """
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        print(f"Created {len(splits)} document chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        
    def query(self, user_query: str) -> str:
        """
        Perform RAG query on loaded documents
        
        Args:
            user_query (str): User's input query
        
        Returns:
            str: Generated response
        """
        if self.vectorstore is None:
            self.prepare_vector_store()
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vectorstore.as_retriever(
                search_kwargs={'k': 3}  # Return top 3 most relevant chunks
            )
        )
        
        # Get response
        response = qa_chain.invoke({"query": user_query})
        return response['result']

def main():
    # Ensure you set GROQ_API_KEY in your environment
    chatbot = RAGChatbot(data_directory='./data')
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        response = chatbot.query(query)
        print("\nResponse:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()