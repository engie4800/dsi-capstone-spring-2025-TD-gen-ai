set backend=ollama
set llm_model_name=mistral
set embed_model_name=nomic-embed-text
set pine_index_name=td-bank-docs-new
set use_cohere=True

streamlit run chatbot.py --server.runOnSave=false
