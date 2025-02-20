from codes.EnchancedResponseChain import EnhancedResponseChain

class EnhancedAgent:
    def __init__(self, secrets):
        self.retriever = EnhancedResponseChain(secrets)

    def invoke(self, query, conversation_history=""):
        return self.retriever.invoke(query, conversation_history)
