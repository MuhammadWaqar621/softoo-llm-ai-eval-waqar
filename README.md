# softoo-llm-ai-eval-waqar
This repository contains the LLM/AI/ML Evaluation Task for Softoo. It includes document search, database querying, and predictive analysis using LLMs. The system integrates FastAPI, LangChain, ChromaDB, and SQL for structured and unstructured data retrieval. Designed for modular task execution and intelligent response aggregation.

# LLM AI Evaluation Framework

This repository contains a framework for evaluating and implementing Large Language Models (LLMs) with specialized capabilities for document search and SQL query generation using Retrieval-Augmented Generation (RAG).

## Features

### Document Search with RAG

- Document ingestion and processing
- Intelligent chunking of documents for optimal context handling
- Vector database integration for efficient similarity search
- Retrieval-Augmented Generation to enhance LLM responses with relevant document context

### SQL Search & Forecasting

- SQL query generation from natural language
- Database schema understanding for accurate query generation
- Forecasting capabilities using SQL data
- RAG-enhanced SQL generation for improved accuracy

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadWaqar621/softoo-llm-ai-eval-waqar.git
   cd softoo-llm-ai-eval-waqar
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

To start the main application:

```bash
python app.py
```

This will launch the service which provides both the document search and SQL generation capabilities.