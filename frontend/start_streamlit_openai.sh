export backend="openai"
export llm_model_name="gpt-4o"
export embed_model_name="text-embedding-ada-002"
export pine_index_name="td-bank-docs-openai"
export use_cohere=True

streamlit run chatbot.py
