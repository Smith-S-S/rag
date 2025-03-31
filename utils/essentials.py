from langchain_ollama import ChatOllama
import yaml
import streamlit as st
import yaml
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_config(config_file= "config/config.yaml"):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def initialize_chat_engine(model_name):
    """Initialize the chat engine with the selected model."""
    return ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.3
    )

def save_uploaded_file(pdf_file_path, uploaded_file):
    file_path = pdf_file_path + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)