import streamlit as st
import os
import base64
import requests
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from streamlit_chat import message

# Groq API details
GROQ_API_KEY = "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"  # Replace with your actual API key
GROQ_MODEL = "llama3-70b-verbose"

st.set_page_config(layout="wide")

persist_directory = "db"

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

@st.cache_resource
def qa_llm():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=None,  # No local model, will call Groq API
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def query_groq_api(prompt):
    url = "https://api.groq.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]}
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error: Unable to fetch response."

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    query = generated_text['result']
    return query_groq_api(query)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        filepath = "docs/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        st.markdown("<h4>Chat Here</h4>", unsafe_allow_html=True)
        user_input = st.text_input("", key="input")

        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]

        if user_input:
            answer = process_answer({'query': user_input})
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(answer)

        if st.session_state["generated"]:
            display_conversation(st.session_state)

if __name__ == "__main__":
    main()
