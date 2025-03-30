import gradio as gr
import os
import argparse
import groq
from rich.console import Console
from rich.panel import Panel
from config import *
from document_search.rag import RAGSystem
from sql_search.sql_database_rag import SQLDatabaseRAG

console = Console()

class UnifiedRAGSystem:
    """Unified RAG system that combines document and SQL RAG outputs."""
    
    def __init__(self, db_connection: str, api_key: str):
        """Initialize both RAG systems and the Groq client."""
        # Initialize document RAG
        self.document_rag = RAGSystem()
        
        # Initialize SQL RAG
        self.sql_rag = SQLDatabaseRAG(db_connection, api_key)
        
        # Initialize Groq client
        self.groq_client = groq.Client(api_key=api_key)
        
        # Initialize chat history
        self.chat_history = []
    
    def get_document_rag_response(self, query: str) -> str:
        """Get response from document RAG system."""
        # Use answer_question instead of query method
        return self.document_rag.answer_question(query)
    
    def get_sql_rag_response(self, query: str) -> str:
        """Get response from SQL RAG system."""
        try:
            # Try with chat history
            return self.sql_rag.process_query(query, self.chat_history)
        except TypeError:
            # If process_query doesn't accept chat_history, try without it
            return self.sql_rag.process_query(query)
    
    def generate_combined_response(self, query: str, doc_response: str, sql_response: str) -> str:
        """Use Groq to generate a combined response from both RAG outputs."""
        # Format the prompt for Groq with instructions for brevity
        prompt = f"""
        You are a concise assistant that provides brief, direct answers.

        USER QUERY: {query}

        DOCUMENT DATABASE INFORMATION: 
        {doc_response}

        SQL DATABASE INFORMATION: 
        {sql_response}

        Instructions:
        1. Provide a direct answer to the query without mentioning your sources
        2. Be extremely concise - keep answers under 300 words maximum
        3. Don't say "Based on the document" or "According to the SQL database"
        4. Focus only on the most relevant information
        5. Skip pleasantries and unnecessary explanations
        6. If there's conflicting information, provide the most accurate answer without explaining the conflict
        """
        
        # Call Groq API
        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Or your preferred Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,  # Reduced to encourage brevity
        )
        
        return response.choices[0].message.content
    
    def process_query(self, query: str) -> str:
        """Process query through both RAG systems and combine results."""
        # Get responses from both systems
        doc_response = self.get_document_rag_response(query)
        sql_response = self.get_sql_rag_response(query)
        
        # Generate combined response using Groq
        final_response = self.generate_combined_response(query, doc_response, sql_response)
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": final_response})
        
        return final_response
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []


# Create and initialize the unified RAG system
def initialize_rag_system(db_connection: str = DB_CONNECTION_STRING, api_key: str = GROQ_API_KEY) -> UnifiedRAGSystem:
    """Initialize the unified RAG system."""
    # Check if vector database exists - use PERSIST_DIRECTORY as in the RAGSystem
    vector_db_exists = os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3"))
    
    # Create unified RAG system
    unified_system = UnifiedRAGSystem(db_connection, api_key)
    
    # Initialize document RAG if needed
    if not vector_db_exists:
        console.print(Panel("Creating vector database by ingesting documents...", title="Initialization"))
        unified_system.document_rag.ingest_documents()
    
    return unified_system


# Gradio interface for the unified RAG system
def create_gradio_interface():
    """Create and launch the Gradio interface."""
    # Initialize the unified RAG system
    system = initialize_rag_system()
    
    # Define CSS for styling
    css = """
    .chatbot-container {
        border-radius: 10px; 
        background-color: #f7f7f8;
    }
    .user-message { 
        background-color: #e6f7ff !important; 
        padding: 10px !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
        text-align: right !important;
    }
    .bot-message { 
        background-color: #f0f0f0 !important; 
        padding: 10px !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
        text-align: left !important;
    }
    .title-container {
        text-align: center;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css) as demo:
        gr.HTML("""
        <div class="title-container">
            <h1>Unified RAG Assistant</h1>
            <p>Ask questions about documents and database data</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3, elem_classes="chatbot-container"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    bubble_full_width=False, 
                    avatar_images=(None, None),
                    show_label=False,
                    height=600,
                    elem_classes=["user-message", "bot-message"]
                )
            
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Type your question here and press Enter...",
                    container=False,
                    scale=8
                )
            
            with gr.Column(scale=1):
                submit_btn = gr.Button("Send", variant="primary")
            
            with gr.Column(scale=1):
                clear_btn = gr.Button("New Chat", variant="secondary")
        
        # State for tracking message history internally
        chat_state = gr.State([])
        
        # Define functions for the interface
        def user_input(message, history):
            # Add user message to history
            return "", history + [[message, None]]
        
        def bot_response(history, chat_state):
            # Get the last user message
            user_message = history[-1][0]
            
            # Process through the unified RAG system
            response = system.process_query(user_message)
            
            # Update history with bot response
            history[-1][1] = response
            
            # Update internal state
            updated_state = chat_state + [(user_message, response)]
            
            return history, updated_state
        
        def clear_chat():
            # Clear the chat history in the system
            system.clear_history()
            
            # Return empty state and history
            return [], []
        
        # Set up event handlers
        msg.submit(
            user_input, 
            [msg, chatbot], 
            [msg, chatbot],
            queue=False
        ).then(
            bot_response, 
            [chatbot, chat_state], 
            [chatbot, chat_state],
            queue=True
        )
        
        submit_btn.click(
            user_input, 
            [msg, chatbot], 
            [msg, chatbot],
            queue=False
        ).then(
            bot_response, 
            [chatbot, chat_state], 
            [chatbot, chat_state],
            queue=True
        )
        
        clear_btn.click(
            clear_chat, 
            None, 
            [chatbot, chat_state],
            queue=False
        )
    
    return demo


# Main entry point
def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified RAG System with Gradio Interface')
    parser.add_argument('--db', type=str, help='Database connection string')
    parser.add_argument('--api-key', type=str, help='Groq API key')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true', help='Create a shareable link')
    args = parser.parse_args()
    
    # Use provided values or defaults from settings
    os.environ["DB_CONNECTION_STRING"] = args.db if args.db else DB_CONNECTION_STRING
    os.environ["GROQ_API_KEY"] = args.api_key if args.api_key else GROQ_API_KEY
    
    console.print(Panel(
        "[bold green]Unified RAG System with Gradio Interface[/bold green]\n\n"
        "Starting Gradio interface, connecting to database, and initializing systems...",
        title="Starting"
    ))
    
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()