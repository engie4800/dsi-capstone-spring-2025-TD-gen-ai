from datetime import datetime

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import cohere
import numpy as np
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
import torch
# import warnings
# warnings.filterwarnings("ignore")


class EnhancedResponseChain:
    def __init__(self, secrets):
        # Initialize Pinecone client with your specific configuration
        pc = Pinecone(api_key=secrets["pinecone_api_key"])

        # Define the environment for your connection
        serverless_spec = ServerlessSpec(
            cloud='gcp',
            region='starter'  # Adjust to match your actual setup, like 'us-west-2' or 'us-east-1' if needed
        )

        self.embeddings = OpenAIEmbeddings()

        # Connect to the 'td-bank-docs' index
        self.index = pc.Index("td-bank-docs", spec=serverless_spec)
        self.vector_store = PineconeVectorStore(self.index, embedding=self.embeddings, text_key="content")
        self.cohere_client = cohere.Client(secrets["openai_api_key"])

        # SPLADE setup
        self.splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.splade_model = AutoModel.from_pretrained("naver/splade-cocondenser-ensembledistil")
        if torch.cuda.is_available():
            self.splade_model = self.splade_model.to('cuda')

        # Create prompt template
        self.prompt_template = """
        You are an AI assistant for TD Bank, tasked with answering questions based on their public documents uploaded to the context. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Conversation history:
        {conversation_history}

        Context:
        {context}

        Question: {question}

        Provide a detailed and well-structured answer, citing specific financial metrics and figures when available:
        """
        PROMPT = PromptTemplate(template=self.prompt_template,
                                input_variables=["conversation_history", "context", "question"])
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.llm_chain = PROMPT | llm

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
        magnitude2 = sum(y ** 2 for y in vec2) ** 0.5
        return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

    def parse_publication_date(self, date_str):
        """Parse publication_date from various formats (e.g., 'YYYY-MM-DD', 'YYYY')."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y").date()
            except ValueError:
                return None

    def get_splade_embedding(self, text):
        inputs = self.splade_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            output = self.splade_model(**inputs)

        sparse_embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        sparse_embedding = normalize(sparse_embedding.reshape(1, -1))[0]
        return sparse_embedding

    def retrieve_candidates(self, query, top_k=100, alpha=0.5):
        """Retrieve initial candidates from Pinecone based on semantic similarity to the query."""
        dense_query = self.embeddings.embed_query(query)
        sparse_query = self.get_splade_embedding(query)

        results = self.index.query(vector=dense_query, top_k=top_k * 2, include_metadata=True)

        scored_results = []
        for match in results["matches"]:
            sparse_doc = match["metadata"].get("sparse_embedding", [])
            if sparse_doc:
                sparse_score = float(np.dot(sparse_query, sparse_doc))
                hybrid_score = alpha * match["score"] + (1 - alpha) * sparse_score
            else:
                hybrid_score = match["score"]
            match["hybrid_score"] = hybrid_score
            scored_results.append(match)

        top_matches = sorted(scored_results, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

        vectors = [
            {
                "summary": match["metadata"].get("summary", "No summary available"),
                "content": match["metadata"].get("content", "No content available"),
                "file_name": match["metadata"].get("file_name", "Unknown file"),
                "page_range": match["metadata"].get("page_range", "Unknown"),
                "publication_date": match["metadata"].get("publication_date", ""),
                "document_type": match["metadata"].get("document_type", ""),
                "vector_score": match["hybrid_score"]
            }
            for match in top_matches
        ]
        return vectors

    def filter_by_filename_similarity(self, vectors, query, threshold=0.5):
        """Filter vectors based on semantic similarity between the query and filenames."""
        query_embedding = self.embeddings.embed_query(query)
        filtered_vectors = []

        for vector in vectors:
            file_name = vector.get("file_name", "Unknown file")
            file_name_embedding = self.embeddings.embed_query(file_name)
            similarity = self.cosine_similarity(query_embedding, file_name_embedding)
            if similarity >= threshold:
                vector["semantic_score"] = similarity
                filtered_vectors.append(vector)

        return sorted(filtered_vectors, key=lambda x: x["semantic_score"], reverse=True)

    def filter_by_metadata(self, vectors, metadata_filter):
        """Filter vectors based on specific metadata such as filenames, document types, or publication dates."""
        filtered_vectors = []
        for vector in vectors:
            matches = all(
                vector.get(key, "").lower() == value.lower()
                for key, value in metadata_filter.items()
                if value
            )
            if matches:
                filtered_vectors.append(vector)
        return filtered_vectors
    
    # New Cohere Reranker Function
    def rerank_with_cohere(self, query, documents, top_n):
        """
        Rerank the given documents with Cohere's rerank API.
        Each document must have a 'content' key to pass to Cohere.
        """
        try:
            results = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[doc["content"] for doc in documents],
                top_n=top_n
            )
            # Return docs in the order given by Cohere
            # result.index refers to the position in the original documents list
            return [documents[result.index] for result in results]
        except Exception as e:
            print(f"Cohere reranking error: {e}")
            # Fall back to top_n from the original documents
            return documents[:top_n]

    def answer_question_with_citations(self, question, conversation_history):
        # Retrieve initial candidate vectors
        candidates = self.retrieve_candidates(question)

        # Step 1: Filter by filename similarity (if filenames are inferred)
        filtered_vectors = self.filter_by_filename_similarity(candidates, query=question)

        # Step 2: Further filter by metadata (if specific metadata is provided or inferred)
        metadata_filter = {
            # This would come from additional logic or metadata reasoning
            "file_name": "",
            "document_type": "",
            "publication_date": "",
        }
        if any(metadata_filter.values()):
            filtered_vectors = self.filter_by_metadata(filtered_vectors, metadata_filter)
        
        # Add rerank step
        reranked_vectors = self.rerank_with_cohere(
            query=question,
            documents=filtered_vectors,
            top_n=10
        )

        # Build context from the filtered results
        # top_context_vectors = filtered_vectors[:10]  # Use only the top 10 for context
        top_context_vectors = reranked_vectors[:10] # shouldn't it already only have the top 10 vectors because of the cohere top_n parameter?
        context = "\n\n".join([
            f"Page {vector['page_range']} (File: {vector['file_name']}): {vector['summary']}"
            for vector in top_context_vectors
        ])

        result = self.llm_chain.run(
            conversation_history=conversation_history,
            context=context,
            question=question
        )

        return result, top_context_vectors

    def invoke(self, question, conversation_history):
        return self.answer_question_with_citations(question, conversation_history)
