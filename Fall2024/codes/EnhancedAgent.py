from codes.EnchancedResponseChain import EnhancedResponseChain
#from codes.EnhancedResponseChain_CohereRerank import EnhancedResponseChain

class EnhancedAgent:
    def __init__(self, secrets,backend, llm_model_name, embed_model_name, pine_index_name, use_cohere):
        self.retriever = EnhancedResponseChain(secrets, backend, llm_model_name, embed_model_name, pine_index_name, use_cohere)

    def invoke(self, query, conversation_history=""):
        return self.retriever.invoke(query, conversation_history)
