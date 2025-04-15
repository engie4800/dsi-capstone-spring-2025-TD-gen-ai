import time
import sys
import json
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

def get_splade_embeddings(text, model, tokenizer, max_length=512, alpha=0.5):
    """Generate SPLADE sparse embeddings for text."""
    tokens = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    
    # Get the logits for each token
    logits = output.logits
    
    # Get the maximum logit for each token
    values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=2)
    
    # Get the non-zero values and their indices
    values = values.squeeze(0)
    indices = torch.nonzero(values > 0).squeeze(1)
    values = values[indices]
    
    # Scale values by alpha
    values = values * alpha
    
    return {
        "indices": indices.tolist(),
        "values": values.tolist()
    }

def chunker(json_file, chunk_size=1, similarity_threshold=0.7, hybrid_alpha=0.0):
    """
    Reads a JSON file (output from pdf_to_text), chunks it by number of pages,
    computes embeddings, merges similar chunks based on cosine similarity, and saves the output to a JSON file.

    Args:
        json_file (str): Path to the JSON file containing page-wise extracted text.
        chunk_size (int): Number of pages per chunk. Default is 1 page.
        similarity_threshold (float): Cosine similarity threshold for merging chunks. Default is 0.7.
        hybrid_alpha (float): Alpha value for hybrid search (0.0 = only dense, 1.0 = only sparse). Default is 0.0.

    Returns:
        dict: A JSON object containing the number of pages processed and the number of resulting chunks.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any other unforeseen errors.
    """
    try:
        print(f"Chunking file {json_file} into {chunk_size}-page chunks using a similarity threshold of {similarity_threshold}. ")
        print(f"Using hybrid search with alpha={hybrid_alpha}")

        # Load the SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load SPLADE model if hybrid search is enabled
        splade_model = None
        splade_tokenizer = None
        if hybrid_alpha > 0:
            splade_model = AutoModelForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')
            splade_tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')

        # Open the JSON file and load page data
        with open(json_file, 'r', encoding='utf-8') as file:
            pages = json.load(file)

        # Ensure the JSON contains a list of pages with "page_content"
        if not isinstance(pages, list) or not all("page_content" in p for p in pages):
            raise ValueError(f"Invalid JSON format in '{json_file}'.")

        # Store total number of pages
        pages_processed = len(pages)

        # Chunk pages together based on chunk_size
        chunks = []
        chunk_pages = []
        for i in tqdm(range(0, len(pages), chunk_size), desc="Chunking pages"):
            chunk_text = " ".join(p["page_content"] for p in pages[i:i + chunk_size])
            chunk_page_range = list(range(i + 1, i + chunk_size + 1))  # Store page numbers
            chunks.append(chunk_text)
            chunk_pages.append(chunk_page_range)

        # Process embeddings while merging similar chunks
        final_chunks = []
        final_pages = []
        final_embeddings = []
        final_sparse_embeddings = []

        for i, chunk in enumerate(chunks):
            # Scale dense embedding by (1 - alpha)
            dense_scale = 1 - hybrid_alpha
            chunk_embedding = model.encode(chunk, convert_to_tensor=True) * dense_scale
            
            # Generate SPLADE sparse embedding if hybrid search is enabled
            sparse_embedding = None
            if hybrid_alpha > 0:
                sparse_embedding = get_splade_embeddings(chunk, splade_model, splade_tokenizer, alpha=hybrid_alpha)

            # If this is the first chunk, add it without checking similarity
            if i == 0:
                final_chunks.append(chunk)
                final_pages.append(chunk_pages[i])
                final_embeddings.append(chunk_embedding)
                if hybrid_alpha > 0:
                    final_sparse_embeddings.append(sparse_embedding)
                continue

            # Compute cosine similarity with the last added chunk
            similarity = util.pytorch_cos_sim(final_embeddings[-1], chunk_embedding).item()

            if similarity >= similarity_threshold:
                # Merge with previous chunk
                final_chunks[-1] += " " + chunk
                final_pages[-1].extend(chunk_pages[i])
                # Recompute embedding for merged chunk
                final_embeddings[-1] = model.encode(final_chunks[-1], convert_to_tensor=True) * dense_scale
                if hybrid_alpha > 0:
                    final_sparse_embeddings[-1] = get_splade_embeddings(final_chunks[-1], splade_model, splade_tokenizer, alpha=hybrid_alpha)
            else:
                # Add as a new chunk
                final_chunks.append(chunk)
                final_pages.append(chunk_pages[i])
                final_embeddings.append(chunk_embedding)
                if hybrid_alpha > 0:
                    final_sparse_embeddings.append(sparse_embedding)

        # Store number of resulting chunks
        num_chunks = len(final_chunks)

        # Prepare output with page numbers
        output_data = []
        for i, (pages, chunk, embedding) in enumerate(zip(final_pages, final_chunks, final_embeddings)):
            chunk_data = {
                "page_range": convert_to_range_string(pages),
                "chunk": chunk,
            }
            if hybrid_alpha > 0:
                chunk_data["sparse_embedding"] = final_sparse_embeddings[i]
            output_data.append(chunk_data)

        # Save to JSON file with "-chunked.json" suffix
        base_name = os.path.splitext(json_file)[0]
        output_filename = f"{base_name}-chunked.json"
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, indent=4, ensure_ascii=False)

        # Return metadata JSON
        return {"pages_processed": pages_processed, "num_chunks": num_chunks}

    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return {"error": f"File '{json_file}' not found."}
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from '{json_file}'.")
        return {"error": f"Failed to parse JSON from '{json_file}'."}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": str(e)}


def convert_to_range_string(num_range):
    if not num_range or len(num_range) == 0:
        return ""
    elif len(num_range) == 1:
        return str(num_range[0])
    else:
        return f"{num_range[0]}-{num_range[-1]}"

# Command-line execution check
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_embedder.py <json_file> [chunk_size] [similarity_threshold] [hybrid_alpha]")
        sys.exit(1)

    json_file = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    similarity_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    hybrid_alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    start_time = time.time()
    result = chunker(json_file, chunk_size, similarity_threshold, hybrid_alpha)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"\nElapsed time {round(elapsed_time)} seconds.\n{result}")

