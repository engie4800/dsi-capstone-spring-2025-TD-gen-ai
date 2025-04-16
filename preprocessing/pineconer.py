import json
import sys
import uuid

import openai
import ollama
import os
import pinecone
from tqdm import tqdm
from langchain_openai import AzureOpenAIEmbeddings
from itertools import islice
from pinecone import ServerlessSpec
from classifier import classifier

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
import numpy as np


def get_splade_embedding(splade_tokenizer,splade_model, text: str) -> np.ndarray:
    """Generate sparse embedding using SPLADE model"""
    # Tokenize and prepare input
    inputs = splade_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # Get model output
    with torch.no_grad():
        output = splade_model(**inputs)

    # Get attention weights and convert to sparse embedding
    attention = output.last_hidden_state.mean(dim=1)  # Average attention across sequence
    sparse_embedding = attention.squeeze().cpu().numpy()

    # Normalize the embedding
    sparse_embedding = normalize(sparse_embedding.reshape(1, -1))[0]

    return sparse_embedding


def generate_embedding(text, embedding_model, embedding_model_name):
    """Generates an embedding for the given text using a pre-instantiated embedding model."""
    try:
        if isinstance(embedding_model, openai.OpenAI):
            response = embedding_model.embeddings.create(
                model=embedding_model_name,
                input=text
            )
            return response.data[0].embedding

        elif isinstance(embedding_model, AzureOpenAIEmbeddings):
            return embedding_model.embed_query(text)

        elif isinstance(embedding_model, str):  # Assume it's an Ollama model name
            response = ollama.embeddings(model=embedding_model_name, prompt=text)
            return response["embedding"]

        else:
            raise ValueError("Invalid embedding model type.")

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def batch(iterable, batch_size):
    """Helper function to yield items in batches."""
    iterable = iter(iterable)
    while True:
        chunk = list(islice(iterable, batch_size))
        if not chunk:
            break
        yield chunk

def pineconer(
    summarized_json_filename,
    from_date,
    to_date,
    backend="ollama",
    embedding_model_name="default",
    pinecone_index_name="td-bank-docs-new",
    pinecone_index_dimension=768,
    batch_size=100
):
    """Reads a summarized JSON file, generates embeddings, and stores them in Pinecone."""
    try:

        secrets_filename = f"secrets-{backend}.json"
        with open(secrets_filename) as f:
            secrets = json.load(f)

        # Instantiate embedding model
        index_dimension = pinecone_index_dimension
        embedding_model = None
        if backend == "openai":
            if embedding_model_name == "default":
                embedding_model_name = "text-embedding-ada-002"
            if embedding_model_name == "text-embedding-ada-002":
                index_dimension = 1536
            embedding_model = openai.OpenAI(api_key=secrets["openai_api_key"])
        elif backend == "azure":
            if embedding_model_name == "default":
                embedding_model_name = "text-embedding-ada-002"
            if embedding_model_name == "text-embedding-ada-002":
                index_dimension = 1536
            embedding_model = AzureOpenAIEmbeddings(
                    model=embedding_model_name,
                    azure_endpoint=secrets["openai_api_endpoint"],
                    api_key=secrets["openai_api_key"],
                    openai_api_version=secrets["openai_api_version"],
            )
        elif backend == "ollama":
            if embedding_model_name == "default":
                embedding_model_name = "nomic-embed-text"
                index_dimension = 768
            embedding_model = embedding_model_name  # Ollama uses the model name directly

        print(f"Embedding summarized JSON file: {summarized_json_filename} using {backend} model {embedding_model_name} to index {pinecone_index_name} with dimension {index_dimension} ")


        # this is pinecone's max
        pinecone_max_metadata_size = 40960



        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=secrets["pinecone_api_key"])

        if pinecone_index_name in pc.list_indexes().names():
            print(f"Index '{pinecone_index_name}' already exists....")
        else:
            pc.create_index(
                name=pinecone_index_name,
                dimension=index_dimension,
                #metric="cosine",
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index = pc.Index(pinecone_index_name)


        # Load summarized JSON file
        with open(summarized_json_filename, 'r', encoding='utf-8') as file:
            chunks = json.load(file)

        json_filename = summarized_json_filename.replace("-summarized", "")
        doc_properties = json.loads(classifier(json_filename))

        records_processed = 0
        processed_chunks = []


        # Initialize SPLADE
        splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        splade_model = AutoModel.from_pretrained("naver/splade-cocondenser-ensembledistil")
        if torch.cuda.is_available():
            splade_model = splade_model.to('cuda')


        for chunk in tqdm(chunks, desc="Processing chunks"):
            page_range = chunk.get("page_range", [])
            chunk_text = chunk.get("chunk", "")
            summary = chunk.get("summary", "")

            embedding_vector = generate_embedding(chunk_text, embedding_model, embedding_model_name)

            sparse_embedding = get_splade_embedding(splade_tokenizer,splade_model,chunk_text)


            if embedding_vector is None:
                print(f"Skipping chunk {page_range} due to embedding error.")
                continue

            # Convert sparse embedding to sparse_values format
            sparse_array = np.array(sparse_embedding)
            indices = np.nonzero(sparse_array)[0].tolist()
            values = sparse_array[indices].tolist()


            record_id = f"content_{str(uuid.uuid4())}"

            # Construct metadata including from_date and to_date
            metadata = {
                "id": record_id,
                "page_range": page_range,
                "content": chunk_text[:40000],
                "summary": summary,
                "date_range": f"{from_date}:{to_date}"
            }

            json_string = json.dumps(metadata)
            metadata_size = sys.getsizeof(json_string)
            if metadata_size > pinecone_max_metadata_size:
                print(f"\nSize of {page_range} chunk metadata exceeds Pinecone limit of {pinecone_max_metadata_size}: {metadata_size} bytes")
                reduction_size = metadata_size - pinecone_max_metadata_size
                summary_len = len(metadata["summary"])
                content_len = len(metadata["content"])
                if summary_len > reduction_size:
                    metadata["summary"] = metadata["summary"][:summary_len-reduction_size]
                    print(f"Reduced summary length from {summary_len} to {len(metadata['summary'])}")
                elif content_len > reduction_size:
                    metadata["content"] = metadata["content"][:content_len-reduction_size]
                    print(f"Reduced content length from {content_len} to {len(metadata['content'])}")

            # Merge additional document properties if available
            metadata.update(doc_properties)

            # Prepare hybrid (dense + sparse) entry for upsert
            processed_chunks.append({
                "id": record_id,
                "values": embedding_vector,
                "sparse_values": {
                    "indices": indices,
                    "values": values
                },
                "metadata": metadata
            })
        # Batch upload embeddings to Pinecone
        for batch_chunk in tqdm(batch(processed_chunks, batch_size), desc="Uploading batches to Pinecone"):
            index.upsert(batch_chunk)
            records_processed += len(batch_chunk)

        return {
            "records_indexed": records_processed,
            "pinecone_index": pinecone_index_name
        }

    except FileNotFoundError:
        print(f"Error: File '{summarized_json_filename}' not found.")
        return {"error": f"File '{summarized_json_filename}' not found."}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": str(e)}

# ✅ Main function for command-line execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed text summaries and store them in a Pinecone index.")

    # Required arguments
    parser.add_argument("summarized_json_filename", type=str, help="Path to the summarized JSON file.")
    parser.add_argument("from_date", type=str, help="Start date in YYYY-MM-DD format.")  # ✅ Required from_date
    parser.add_argument("to_date", type=str, help="End date in YYYY-MM-DD format.")  # ✅ Required to_date

    # Optional arguments
    parser.add_argument("--embedding_model_name", type=str, default="default", help="Name of the embedding model to use.")
    parser.add_argument("--pinecone_index_name", type=str, default="td-bank-docs-new", help="Name of the Pinecone index.")
    parser.add_argument("--backend", type=str, default="ollama", choices=["openai", "ollama", "azure"], help="Embedding backend: 'openai', 'ollama', or 'azure'.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of records to process per batch.")

    args = parser.parse_args()

    # Run the pineconer function with the provided arguments
    result = pineconer(
        summarized_json_filename=args.summarized_json_filename,
        from_date=args.from_date,
        to_date=args.to_date,
        backend=args.backend,
        embedding_model_name=args.embedding_model_name,
        pinecone_index_name=args.pinecone_index_name,
        batch_size=args.batch_size
    )

    print(json.dumps(result, indent=4))
