import time
import json
import os
import argparse
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI


def summarizer(input_file, backend="ollama", modelname="default", hybrid_alpha=0.0):
    """
    Determines which backend model to use for summarizing the input file and calls the corresponding function.

    Args:
        input_file (str): Path to the input JSON file containing chunked text.
        backend (str, optional): Specifies which backend model to use.
                                 Options: 'ollama' (default), 'openai', or 'azure'.
        modelname (str, optional): Name of the model to use. Default is 'default'.
        hybrid_alpha (float, optional): Alpha value for hybrid search (0.0 = only dense, 1.0 = only sparse).
                                      Default is 0.0.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    if backend in ["openai", "azure"]:
        return summarize_text_with_backend(backend, input_file,
                                    model_name="gpt-4o" if modelname == "default"
                                    else modelname,
                                    hybrid_alpha=hybrid_alpha)
    elif backend in ["ollama"]:
        return summarize_text_with_backend(backend, input_file,
                                    model_name="mistral" if modelname == "default"
                                    else modelname,
                                    hybrid_alpha=hybrid_alpha)
    else:
        raise ValueError(f"Unknown backend: {backend}, must be 'ollama', 'azure', or 'openai'")


def summarize_text_with_backend(backend, input_file, model_name="mistral", max_summary_tokens=1024, temp=0, hybrid_alpha=0.0):
    """
    Loads the appropriate API credentials and model configuration for the specified backend and
    calls the summarization function.

    Args:
        backend (str): The backend to use for summarization ('ollama', 'openai', 'azure').
        input_file (str): Path to the input JSON file containing chunked text.
        model_name (str, optional): The model to use for text summarization. Defaults to 'mistral'.
        max_summary_tokens (int, optional): Maximum number of tokens for the summary. Defaults to 1024.
        temp (int, optional): Temperature setting for the LLM (controls randomness in responses). Defaults to 0.
        hybrid_alpha (float, optional): Alpha value for hybrid search (0.0 = only dense, 1.0 = only sparse).
                                      Default is 0.0.

    Returns:
        dict: A JSON object containing metadata such as the number of chunks processed and output file name.

    Raises:
        Exception: If an error occurs while loading credentials or initializing the model.
    """
    try:
        print(f"Summarizing text from {input_file} using backend {backend} and model name {model_name}")
        print(f"Using hybrid search with alpha={hybrid_alpha}")

        # Load the API key info
        secrets_filename = f"secrets-{backend}.json"
        with open(secrets_filename) as f:
            secrets = json.load(f)

        # Initialize the LLM model based on backend
        if backend == "azure":
            llm = AzureChatOpenAI(
                model=model_name,
                temperature=temp,
                max_retries=2,
                max_tokens=max_summary_tokens,
                azure_endpoint=secrets["openai_api_endpoint"],
                api_key=secrets["openai_api_key"],
                openai_api_version=secrets["openai_api_version"]
            )
        else:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temp,
                max_retries=2,
                max_tokens=max_summary_tokens,
                api_key=secrets["openai_api_key"],
                openai_api_base=secrets["openai_api_endpoint"],
            )

        return summarize_text_with_llm(input_file, llm, hybrid_alpha)

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return {"error": str(e)}


def summarize_text_with_llm(input_file, llm, hybrid_alpha=0.0):
    """
    Reads a JSON file containing chunked text, generates summaries using an LLM,
    and writes the summarized output to a new JSON file.

    Args:
        input_file (str): Path to the JSON file containing chunked text.
        llm (ChatOpenAI or AzureChatOpenAI): The initialized language model.
        hybrid_alpha (float): Alpha value for hybrid search (0.0 = only dense, 1.0 = only sparse). Default is 0.0.

    Returns:
        dict: A JSON object containing the number of processed chunks and the output file name.

    Raises:
        FileNotFoundError: If the specified input file does not exist.
        json.JSONDecodeError: If the input file is not a valid JSON file.
        Exception: For any other unforeseen errors.
    """
    try:
        # Load the JSON file
        with open(input_file, 'r', encoding='utf-8') as file:
            chunks = json.load(file)

        # Ensure valid format
        if not isinstance(chunks, list) or not all("chunk" in chunk for chunk in chunks):
            raise ValueError(f"Invalid JSON format in '{input_file}'.")

        # Generate summaries for each chunk
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Preserve any existing sparse embeddings if present
            sparse_embedding = chunk.get("sparse_embedding", None)
            
            results = get_chunk_info(llm, chunk["chunk"])
            chunk["summary"] = results["summary"]
            
            # Restore sparse embedding if it existed and hybrid search is enabled
            if sparse_embedding and hybrid_alpha > 0:
                # Scale the sparse embedding values by alpha
                sparse_embedding["values"] = [v * hybrid_alpha for v in sparse_embedding["values"]]
                chunk["sparse_embedding"] = sparse_embedding

        # Generate output filename with "-summarized.json" suffix
        base_name = os.path.splitext(input_file)[0].replace("-chunked", "")
        output_filename = f"{base_name}-summarized.json"

        # Save the summarized chunks
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            json.dump(chunks, output_file, indent=4, ensure_ascii=False)

        return {"chunks_processed": len(chunks), "output_file": output_filename}

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return {"error": f"File '{input_file}' not found."}
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from '{input_file}'.")
        return {"error": f"Failed to parse JSON from '{input_file}'."}
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return {"error": str(e)}


def get_chunk_info(llm, chunk_text):
    """
    Uses a language model (LLM) to generate a summary for a given text chunk.

    Args:
        llm (ChatOpenAI or AzureChatOpenAI): The initialized language model.
        chunk_text (str): The text content of a chunk.

    Returns:
        dict: A JSON object containing the summary of the chunk.

    Raises:
        Exception: If the LLM fails to generate a summary.
    """
    try:
        messages = [
            {"role": "system",
             "content": "You are a financial text analysis AI. If you don't know the answer, say so."},
            {"role": "user", "content":
                f"""
                Read the following financial text and return a concise summary of the main insights.

                Here is the text:
                {chunk_text}
                """}
        ]

        response = llm.invoke(messages)

        # Extract response content
        generated_text = response.content.strip()

        return {"summary": generated_text}

    except Exception as e:
        print(f"\nError generating summary: {e}")
        return {"summary": "Unknown Summary"}


# Command-line execution check
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize chunked text using LLMs.")

    # Required argument: input file
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")

    # Optional arguments
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "openai", "azure"],
                        help="Specify the backend model to use (default: 'ollama').")
    parser.add_argument("--llm_model_name", type=str, default="default",
                        help="Specify the LLM model name (default: 'default').")
    parser.add_argument("--hybrid_alpha", type=float, default=0.0,
                        help="Alpha value for hybrid search (0.0 = only dense, 1.0 = only sparse). Default is 0.0.")

    args = parser.parse_args()

    # ✅ Call summarizer with parsed arguments
    start_time = time.time()
    result = summarizer(args.input_file, backend=args.backend, modelname=args.llm_model_name, hybrid_alpha=args.hybrid_alpha)
    end_time = time.time()

    # ✅ Display result with elapsed time
    elapsed_time = round(end_time - start_time)
    print(f"\nElapsed time {elapsed_time} seconds.\n{result}")