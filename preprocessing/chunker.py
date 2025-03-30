import time
import sys
import json
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def chunker(json_file, chunk_size=1, similarity_threshold=0.7):
    """
    Reads a JSON file (output from pdf_to_text), chunks it by number of pages,
    computes embeddings, merges similar chunks based on cosine similarity, and saves the output to a JSON file.

    Args:
        json_file (str): Path to the JSON file containing page-wise extracted text.
        chunk_size (int): Number of pages per chunk. Default is 1 page.
        similarity_threshold (float): Cosine similarity threshold for merging chunks. Default is 0.7.

    Returns:
        dict: A JSON object containing the number of pages processed and the number of resulting chunks.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any other unforeseen errors.
    """
    try:
        print(f"Chunking file {json_file} into {chunk_size}-page chunks using a similarity threshold of {similarity_threshold}. ")

        # Load the SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

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

        for i, chunk in enumerate(chunks):
            chunk_embedding = model.encode(chunk, convert_to_tensor=True)

            # If this is the first chunk, add it without checking similarity
            if i == 0:
                final_chunks.append(chunk)
                final_pages.append(chunk_pages[i])
                final_embeddings.append(chunk_embedding)
                continue

            # Compute cosine similarity with the last added chunk
            similarity = util.pytorch_cos_sim(final_embeddings[-1], chunk_embedding).item()

            if similarity >= similarity_threshold:
                # Merge with previous chunk
                final_chunks[-1] += " " + chunk
                final_pages[-1].extend(chunk_pages[i])
                # Recompute embedding for merged chunk
                final_embeddings[-1] = model.encode(final_chunks[-1], convert_to_tensor=True)
            else:
                # Add as a new chunk
                final_chunks.append(chunk)
                final_pages.append(chunk_pages[i])
                final_embeddings.append(chunk_embedding)

        # Store number of resulting chunks
        num_chunks = len(final_chunks)

        # Prepare output with page numbers
        output_data = [
            {
                "page_range": convert_to_range_string(pages),
                "chunk": chunk,
             #"summary":"",
             #"embedding": embedding.tolist()
             }
            for pages, chunk, embedding in zip(final_pages, final_chunks, final_embeddings)
        ]

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
        print("Usage: python text_embedder.py <json_file> [chunk_size] [similarity_threshold]")
        sys.exit(1)

    json_file = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    similarity_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    start_time = time.time()
    result = chunker(json_file, chunk_size, similarity_threshold)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"\nElapsed time {round(elapsed_time)} seconds.\n{result}")

