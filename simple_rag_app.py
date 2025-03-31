import streamlit as st
import re

# extracts text from PDF files so that you can work with it in your code.
from langchain_community.document_loaders import PDFPlumberLoader 

# Text splitting is useful when you have long documents (like the ones loaded from PDFs).
# RecursiveCharacterTextSplitter is a tool that splits the text into smaller parts based on character count 
# while preserving the logical structure of the document (e.g., splitting sentences or paragraphs neatly)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector stores are used to store vectors, which are numeric representations of 
# words, sentences, or entire documents. 
# These vectors are used for searching or matching content based on meaning.

# ==> InMemoryVectorStore is a place in memory where you can store these vectors temporarily, 
# so you can quickly look up and compare similar documents or text. It’s like a mini-database but stored in RAM.
from langchain_core.vectorstores import InMemoryVectorStore

# words to --> related numbers (vercors) using a pre-trained model.[ollama]
from langchain_ollama import OllamaEmbeddings

# helps us to structure the conversation between the user and the AI.
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from app_setup import *
from essentials import *

config = load_config()
# Now, you can access the settings like this:
PDF_STORAGE_PATH = config['pdf_storage_path']
EMBEDDING_MODEL = OllamaEmbeddings(model=config['embedding_model'])
LANGUAGE_MODEL = OllamaLLM(model=config['language_model'])
thinking_enabled = config["thinking"]


PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    # print("response: ",response)

    # Clean the response by removing any <think> tags if thinking is disabled
    if not thinking_enabled:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    return response


def main():
    setup_custom_styling_rag_app()
    # UI Configuration
    st.title("📘 DocuMind AI")
    st.markdown("### Your Intelligent Document Assistant")
    st.markdown("---")

    # File Upload Section
    uploaded_pdf = st.file_uploader(
        "Upload Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

    if uploaded_pdf:
        # process the uploaded PDF files
        saved_path = save_uploaded_file(PDF_STORAGE_PATH, uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)

        st.success("✅ Document processed successfully! Ask your questions below.")
        user_input = st.chat_input("Enter your question about the document...")

        # start the conversation
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)

if __name__ == "__main__":
    main()