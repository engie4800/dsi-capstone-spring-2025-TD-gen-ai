from codes.EnchancedResponseChain import EnhancedResponseChain #this import is using the original enhancedresponsechain instead of the cohere one

class EnhancedAgent:
    def __init__(self, secrets):
        self.retriever = EnhancedResponseChain(secrets)

    def invoke(self, query, conversation_history=""):
        return self.retriever.invoke(query, conversation_history)
