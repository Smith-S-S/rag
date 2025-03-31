import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from app_setup import *
from essentials import *


def get_system_prompt():
    """Get the system prompt template for the AI."""
    return SystemMessagePromptTemplate.from_template(
        "You are an expert AI coding assistant. Provide concise, correct solutions "
        "with strategic print statements for debugging. Always respond in English."
    )

def initialize_session_state():
    """Initialize the session state for chat history."""
    if "message_log" not in st.session_state:
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

def display_chat_messages():
    """Display all chat messages in the chat container."""
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def build_prompt_chain():
    """Build the prompt chain from the chat history."""

    """
    Takes all your previous messages
    Prepares them in a way the AI can understand
    Makes sure the AI remembers the context of your conversation
    """
    prompt_sequence = [get_system_prompt()]
    for msg in st.session_state.message_log:
        print("message: ",msg)
        if msg["role"] == "user":
            print("msg['role']",msg["role"])
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            print("msg['role']=ai",msg["role"])
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain, llm_engine):
    """Generate AI response using the prompt chain and LLM engine."""
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def process_user_input(user_query, llm_engine):
    """Process user input and generate AI response."""
    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
        
        with st.spinner("🧠 Processing..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain, llm_engine)
        
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.rerun()

def main():
    """Main function to run the Streamlit app."""
    # Setup
    setup_custom_styling_chat_bot()
    st.title("🧠 DeepSeek Code Companion")
    st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")
    
    # Initialize components
    selected_model = setup_sidebar_chat_bot()
    llm_engine = initialize_chat_engine(selected_model)
    initialize_session_state()
    
    # Display chat interface
    display_chat_messages()
    
    # Handle user input
    user_query = st.chat_input("Type your coding question here...")
    process_user_input(user_query, llm_engine)

if __name__ == "__main__":
    main()