
from codes.EnhancedAgent import EnhancedAgent
import os
import json


# code block required because: it disables Streamlit's "hot reload" module path inspection
# for anything under the torch package — which avoids triggering the faulty dynamic inspection logic.
###### BEGIN: PATCHES for streamlit-torch incompatibilities
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch transformers"
import streamlit as st
import torch

# Prevent access to __path__ from triggering instantiation
# this is really the one that resolved the issue
if hasattr(torch, "classes"):
    class SafePath:
        def __getattr__(self, item):
            return []

    torch.classes.__path__ = SafePath()
###### END: PATCHES for streamlit-torch incompatibilities

def write_no_latex(*args, **kwargs):
    """Custom st.write() that disables LaTeX rendering."""
    escaped_args = [
        str(arg).replace("$", "\\$") if isinstance(arg, str) else arg
        for arg in args
    ]
    
    # Join all arguments with spaces (mimicking st.write behavior)
    text = " ".join(map(str, escaped_args))
    st.markdown(text,unsafe_allow_html=True)   


def chatter(trace_on=True,
            backend="ollama",
            llm_model_name="default",
            embed_model_name="default",
            pine_index_name="td-bank-docs-new",
            use_cohere=True,
            hybrid_alpha=0,
            confidence_threshold=0
            ):
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


    if (trace_on):
        print(f"Tracing is on: Chatting using backend {backend} with llm {llm_model_name}, embed {embed_model_name}, Pinecone index {pine_index_name}, use_cohere {use_cohere}, confidence_threshold {confidence_threshold}, hybrid_alpha {hybrid_alpha}")

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
    agent = EnhancedAgent(secrets,
                          trace_on=trace_on,
                          backend=backend,
                          llm_model_name=llm_model_name,
                          embed_model_name=embed_model_name,
                          pine_index_name=pine_index_name,
                          use_cohere=use_cohere,
                          hybrid_alpha=hybrid_alpha,
                          confidence_threshold=confidence_threshold)

    # Chatbot Application
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("TD Bank Performance Chat Assistant")



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
                answer, documents,confidence_score, reason = agent.invoke(user_question, conversation_history)

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
    #use_cohere = eval(os.getenv("use_cohere", "True"))


    # Sidebar with reset button
    with st.sidebar:
        st.title("Chat Options")

        # Checkbox to toggle Cohere reranking
        trace_exec = st.checkbox("Trace On", value=False)

        # Checkbox to toggle Cohere reranking
        use_cohere = st.checkbox("Use Cohere Reranker", value=False)

        use_confidence_threshold = st.checkbox("Use Confidence Threshold", value=False)
        confidence_threshold = st.number_input(
            "Confidence Threshold (0.0 to 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            disabled=not use_confidence_threshold
        )
        # Force confidence_threshold to 0.0 if hybrid search is off
        if not use_confidence_threshold:
            confidence_threshold = 0.0
        confidence_threshold = round(confidence_threshold, 2)

        use_hybrid = st.checkbox("Use Hybrid Search", value=False)
        hybrid_alpha = st.number_input(
            "Hybrid Alpha (0.0 to 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            disabled=not use_hybrid
        )
        # Force hybrid_alpha to 0.0 if hybrid search is off
        if not use_hybrid:
            hybrid_alpha = 0.0
        hybrid_alpha = round(hybrid_alpha, 2)

        if st.button("Start New Chat"):
            st.session_state.messages = []
            st.rerun()

    chatter(trace_exec, backend, llm_model_name, embed_model_name, pine_index_name,
            use_cohere, hybrid_alpha, confidence_threshold
            )


# Entry point
if __name__ == "__main__":
    main()
