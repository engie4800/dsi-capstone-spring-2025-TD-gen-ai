import streamlit as st
from codes.EnhancedAgent import EnhancedAgent
import os
import json
import argparse


def write_no_latex(*args, **kwargs):
    """Custom st.write() that disables LaTeX rendering."""
    text = str(args[0])
    st.markdown(f"<pre>{text}</pre>", unsafe_allow_html=True)

def chatter(backend="ollama", llm_model_name="default", embed_model_name="default", pine_index_name="td-bank-docs-new"):
    """
    Function to run the chatbot application.
    
    Args:
        backend (str): Backend for LLM (ollama, openai, azure).
        llm_model_name (str): Name of the LLM model.
        embed_model_name (str): Name of the embedding model.
        pine_index_name (str): Name of the Pinecone index.
    """

    # Set default model names based on backend
    if backend in ["openai", "azure"]:
        if llm_model_name == "default":
            llm_model_name="gpt-4o"
        if embed_model_name == "default":
            embed_model_name="text-embedding-ada-002"
    elif backend in ["ollama"]:
        if llm_model_name == "default":
            llm_model_name="mistral"
        if embed_model_name == "default":
            embed_model_name="nomic-embed-text"        
    else:
        raise ValueError(f"Unknown backend: [{backend}], must be 'ollama', 'azure', or 'openai'")



    print(f"Chatting using backend {backend} with llm {llm_model_name}, embed {embed_model_name}, and Pinecone index {pine_index_name}...")

    # Load secrets info from appropriate backend secrets file
    secrets_filename = f"secrets_{backend}.json"
    with open(secrets_filename) as f:
        secrets = json.load(f)

    # Set environment variables
    os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
    os.environ["LANGCHAIN_API_KEY"] = secrets["langchain_api_key"]
    #os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "TD-LLM"
    os.environ["PINECONE_API_KEY"] = secrets["pinecone_api_key"]

    # Override st.write globally to disable LaTeX rendering
    st.write = write_no_latex

    # Initialize agent with secrets
    agent = EnhancedAgent(secrets, backend=backend, llm_model_name=llm_model_name, embed_model_name=embed_model_name, pine_index_name=pine_index_name)

    # Chatbot Application
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("TD Bank Performance Chat Assistant")

    # Sidebar with reset button
    with st.sidebar:
        st.title("Chat Options")
        if st.button("Start New Chat"):
            st.session_state.messages = []
            st.rerun()

    # Main chat container
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about TD Bank's performance")

    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                conversation_history = "\n".join(
                    f"Q: {msg['content']}" if msg['role'] == 'user' else f"A: {msg['content']}"
                    for msg in st.session_state.messages
                )
                answer, documents = agent.invoke(user_question, conversation_history)

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": documents
                })

                # Display assistant's answer
                st.write(answer)

                # Show sources in an expandable section
                with st.expander("View Sources"):
                    for i, doc in enumerate(documents, 1):
                        st.write(f"**Source {i}:** Page {doc['page_range']} (File: {doc['file_name']})")
                        st.write(doc['summary'])

def main():
    """
    Main function to parse command-line arguments and run the chatbot.
    """
    # streamlit doesn't support command-line arguments, so we use environment variables instead
    backend = os.getenv("backend", "ollama")
    llm_model_name = os.getenv("llm_model_name", "default")
    embed_model_name = os.getenv("embed_model_name", "default")
    pine_index_name = os.getenv("pine_index_name", "td-bank-docs-new")

    chatter(backend, llm_model_name, embed_model_name, pine_index_name)


# Entry point
if __name__ == "__main__":
    main()
