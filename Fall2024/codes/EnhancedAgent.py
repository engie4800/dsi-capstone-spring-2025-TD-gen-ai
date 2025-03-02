from codes.EnchancedResponseChain import EnhancedResponseChain

class EnhancedAgent:
    def __init__(self, secrets,backend, llm_model_name, embed_model_name, pine_index_name):
        self.retriever = EnhancedResponseChain(secrets, backend, llm_model_name, embed_model_name, pine_index_name)

    def invoke(self, query, conversation_history=""):
        return self.retriever.invoke(query, conversation_history)
