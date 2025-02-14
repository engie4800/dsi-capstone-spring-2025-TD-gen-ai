import streamlit as st
from codes.EnhancedAgent import EnhancedAgent
import os
import json

with open("secrets.json") as f:
     secrets = json.load(f)

#secrets = {
#    "openai_api_key": st.secrets["openai_api_key"],
#    "langchain_api_key": st.secrets["langchain_api_key"],
#    "unstructured_api_key": st.secrets["unstructured_api_key"],
#    "huggingface_api_key": st.secrets["huggingface_api_key"],
#    "pinecone_api_key": st.secrets["pinecone_api_key"]
#}

os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
os.environ["LANGCHAIN_API_KEY"] = secrets["langchain_api_key"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "TD-LLM"
#os.environ["UNSTRUCTURED_API_KEY"] = secrets["unstructured_api_key"]
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets["huggingface_api_key"]
os.environ["PINECONE_API_KEY"] = secrets["pinecone_api_key"]

agent = EnhancedAgent(secrets)

# Chatbot Application

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("TD Bank Performance Chat Assistant")

# Sidebar
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
    # Add user message to chat
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
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": documents
            })
            st.write(answer)
            # Add source details in an expandable section
            with st.expander("View Sources"):
                for i, doc in enumerate(documents, 1):
                    st.write(f"**Source {i}:** Page {doc['page_range']} "
                             f"(File: {doc['file_name']})")
                    st.write(doc['summary'])
