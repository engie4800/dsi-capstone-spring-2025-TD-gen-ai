import ollama

from langchain.embeddings.base import Embeddings


class OllamaEmbeddings(Embeddings):
    """Wrapper class to use Ollama embeddings in LangChain-compatible vector store."""
    
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def embed_query(self, text):
        """Generate an embedding for a single query."""
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

    def embed_documents(self, texts):
        """Generate embeddings for a list of texts."""
        return [self.embed_query(text) for text in texts]
