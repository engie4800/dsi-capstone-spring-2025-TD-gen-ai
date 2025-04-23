import cohere
import torch
from datetime import datetime
from transformers import AutoModelForMaskedLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from .OllamaEmbeddings import OllamaEmbeddings


# import warnings

# warnings.filterwarnings("ignore")

class EnhancedResponseChain:
    def __init__(self, secrets, trace_on, backend, llm_model_name, embed_model_name, pine_index_name, use_cohere,
                 hybrid_alpha=0.5):
        """
        Initialize the EnhancedResponseChain with hybrid search support.

        Args:
            secrets: Dictionary containing API keys and secrets
            backend: Backend to use ('ollama', 'azure', 'openai')
            llm_model_name: Name of the LLM model to use
            embed_model_name: Name of the embedding model to use
            pine_index_name: Name of the Pinecone index
            use_cohere: Whether to use Cohere for reranking
            hybrid_alpha: Alpha value for hybrid search (0.0 = only dense, 1.0 = only sparse)
        """
        self.trace_on = trace_on
        # Initialize SPLADE model if hybrid search is enabled
        self.hybrid_alpha = hybrid_alpha
        if hybrid_alpha > 0:
            self.splade_model = AutoModelForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')
            self.splade_tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')

        # Initialize Pinecone client with your specific configuration
        pc = Pinecone(api_key=secrets["pinecone_api_key"])

        # Define the environment for your connection
        serverless_spec = ServerlessSpec(
            cloud='gcp',
            region='starter'  # Adjust to match your actual setup, like 'us-west-2' or 'us-east-1' if needed
        )

        if backend in ["openai"]:
            self.embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(
                model=llm_model_name,
                temperature=0,
                api_key=secrets["openai_api_key"],
                openai_api_base=secrets["openai_api_endpoint"],
            )
        elif backend in ["azure"]:
            self.embeddings = AzureOpenAIEmbeddings(
                model=embed_model_name,
                azure_endpoint=secrets["openai_api_endpoint"],
                api_key=secrets["openai_api_key"],
                openai_api_version=secrets["openai_api_version"],
            )
            llm = AzureChatOpenAI(
                model=llm_model_name,
                temperature=0,
                max_retries=2,
                azure_endpoint=secrets["openai_api_endpoint"],
                api_key=secrets["openai_api_key"],
                openai_api_version=secrets["openai_api_version"]
            )
        elif backend in ["ollama"]:
            self.embeddings = OllamaEmbeddings(model_name=embed_model_name)
            llm = ChatOpenAI(
                model=llm_model_name,
                temperature=0,
                api_key=secrets["openai_api_key"],
                openai_api_base=secrets["openai_api_endpoint"],
            )
        else:
            raise ValueError(f"Unknown backend: {backend}, must be 'ollama', 'azure', or 'openai'")

        # Connect to the 'td-bank-docs' index
        self.index = pc.Index(pine_index_name, spec=serverless_spec)
        self.vector_store = PineconeVectorStore(self.index, embedding=self.embeddings, text_key="content")
        self.cohere_client = cohere.Client(secrets["cohere_api_key"])
        self.use_cohere = use_cohere

        # Create prompt template
        self.prompt_template = """
        You are an AI assistant for TD Bank, tasked with answering questions based on their public documents 
        from 2020-11-01 to 2025-01-31 
        uploaded to the context. 
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

        self.llm_chain = PROMPT | llm

    def get_splade_embedding(self, text, max_length=512):
        """Generate SPLADE sparse embedding for text."""
        tokens = self.splade_tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        with torch.no_grad():
            output = self.splade_model(**tokens)

        logits = output.logits
        values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=2)
        values = values.squeeze(0)
        # the line below was causing runtime errors, th2 lines afterwards seemed to fix this
        #indices = torch.nonzero(values > 0).squeeze(1)
        mask = (values > 0)
        indices = mask.nonzero(as_tuple=True)[0]

        values = values[indices]

        return {
            "indices": indices.tolist(),
            "values": values.tolist()
        }

    # @torch.no_grad()
    # def get_splade_embedding(text, tokenizer, model, max_length=512):
    #     tokens = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    #     with torch.no_grad():
    #         output = model(**tokens)
    #
    #     logits = output.logits
    #     values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=2)
    #     values = values.squeeze(0)
    #     indices = torch.nonzero(values > 0).squeeze(1)
    #     values = values[indices]
    #
    #     return {
    #         "indices": indices.tolist(),
    #         "values": values.tolist()
    #     }

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

    def retrieve_candidates(self, query, top_k=100):
        """Retrieve initial candidates from Pinecone based on hybrid search (dense + sparse)."""
        query_embedding = self.embeddings.embed_query(query)

        if self.hybrid_alpha > 0:
            if self.trace_on:
                print(f"Hybrid search enabled with alpha = {self.hybrid_alpha}")
            # Get SPLADE sparse vector
            sparse = self.get_splade_embedding(query)

            # Normalize SPLADE vector to match Pinecone format
            sparse_vector = {
                "indices": sparse["indices"],
                "values": sparse["values"]
            }

            results = self.index.query(
                vector=query_embedding,
                sparse_vector=sparse_vector,
                top_k=top_k,
                include_metadata=True,
                alpha=self.hybrid_alpha  # weight for sparse component
            )
        else:
            if self.trace_on:
                print("Using dense search")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

        vectors = [
            {
                "summary": match["metadata"].get("summary", "No summary available"),
                "content": match["metadata"].get("content", "No content available"),
                "file_name": match["metadata"].get("file_name", "Unknown file"),
                "page_range": match["metadata"].get("page_range", "Unknown"),
                "publication_date": match["metadata"].get("publication_date", ""),
                "document_type": match["metadata"].get("document_type", ""),
                "vector_score": match["score"],
            }
            for match in results["matches"]
        ]
        return vectors

    def filter_by_filename_similarity(self, vectors, query, threshold=0.4):
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
            return [documents[result.index] for result in results.results]
        except Exception as e:
            print(f"Cohere reranking error: {e}")
            # Fall back to top_n from the original documents
            return documents[:top_n]

    def answer_question_with_citations(self, question, conversation_history):
        # Retrieve initial candidate vectors
        candidates = self.retrieve_candidates(question)
        if self.trace_on:
            print(f"Retrieved {len(candidates)} candidates from Pinecone.")

        # Step 1: Filter by filename similarity (if filenames are inferred)
        filtered_vectors = self.filter_by_filename_similarity(candidates, query=question)

        if self.trace_on:
            print(f"Filtered down to {len(filtered_vectors)} candidates based on filename similarity")

        # Step 2: Further filter by metadata (if specific metadata is provided or inferred)
        metadata_filter = {
            # This would come from additional logic or metadata reasoning
            "file_name": "",
            "document_type": "",
            "publication_date": "",
        }
        if any(metadata_filter.values()):
            print(f"Further filtering based on metadata: {metadata_filter}")
            filtered_vectors = self.filter_by_metadata(filtered_vectors, metadata_filter)

        if self.use_cohere:
            # Add rerank step
            reranked_vectors = self.rerank_with_cohere(
                query=question,
                documents=filtered_vectors,
                top_n=10
            )
            top_context_vectors = reranked_vectors[:10]
            context = "\n\n".join([
                f"Page {vector['page_range']} (File: {vector['file_name']}): {vector['summary']}"
                for vector in top_context_vectors
            ])
            if self.trace_on:
                print(f"Reranked down to {len(reranked_vectors)} candidates using Cohere.")
        else:
            # Build context from the filtered results
            top_context_vectors = filtered_vectors[:10]  # Use only the top 10 for context
            context = "\n\n".join([
                f"Page {vector['page_range']} (File: {vector['file_name']}): {vector['summary']}"
                for vector in top_context_vectors
            ])

            # print(f"Using the following context for the question:\n{context}")

        # escaped_context = context.replace("$", "\\$")
        # print(f"Escaped context: {escaped_context}")

        response = self.llm_chain.invoke({
            "conversation_history": conversation_history,
            "context": context,
            "question": question
        })
        # escaped_response = response.content.replace("$", "\\$")
        # print(f"Generated response: {response.content}")

        result = response.content
        # result = escaped_response

        return result, top_context_vectors

    def invoke(self, question, conversation_history):
        return self.answer_question_with_citations(question, conversation_history)