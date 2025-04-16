import json
import os
import re
import nltk
import sys
import time
import tqdm
from nltk.tokenize import sent_tokenize

from langchain_community.document_loaders import PyPDFLoader


# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
# print(nltk.data.find('tokenizers/punkt'))
# from nltk.tokenize import sent_tokenize
#
# text = "This is a test. Here's another sentence."
# print(sent_tokenize(text))


def texter(pdf_path, out_filename=None, text_only=False):
    """
    Converts a PDF file to text and saves it to an output file. Also extracts PDF properties
    and saves them to a JSON file.

    Parameters:
    pdf_path (str): The path to the PDF file to be converted.
    output_path (str, optional): The path to the output text file. If not provided,
                                 the output file will be created in the same directory
                                 as the PDF file with the same base name and a .txt extension.

    Returns:
    str: The path to the output text file.
    """
    try:
        # Save text to output file
        base_name = os.path.splitext(pdf_path)[0]
        if out_filename is None:
            if text_only:
                out_filename = f"{base_name}.txt"
            else:
                out_filename = f"{base_name}.json"



        print(f"Converting PDF file {pdf_path} to file {out_filename}...")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        #text = "\n".join(doc.page_content for doc in documents)
        # Extract pages as JSON objects
        #pages_json = [{"page_number": i + 1, "page_content": clean_string(doc.page_content)} for i, doc in enumerate(documents)]
        num_pages = 0
        if text_only:
            text = "\n".join(clean_string(doc.page_content) for doc in documents)
            #print(text)
            num_pages = len(documents)
            with open(out_filename, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
        else:
            pages_json = []
            for i, doc in enumerate(tqdm.tqdm(documents, desc="Processing documents")):
                page = {
                    "page_number": i + 1,
                    "page_content": clean_string(doc.page_content)
                }
                pages_json.append(page)
            num_pages = len(pages_json)
            #with open(out_filename, 'w', encoding='utf-8') as output_file:
            #    output_file.write(text)

            # Save page content to JSON file
            with open(out_filename, 'w', encoding='utf-8') as output_file:
                json.dump(pages_json, output_file, indent=4, ensure_ascii=False)

        # Save PDF properties to JSON file

        # Extract PDF properties from the first document's metadata
        if documents:
            pdf_properties = documents[0].metadata
        else:
            pdf_properties = {}

        # Convert PDF properties to a serializable format
        serializable_properties = {key: str(value) for key, value in pdf_properties.items()}


        prop_filename = f"{base_name}-prop.json"
        with open(prop_filename, 'w', encoding='utf-8') as prop_file:
            json.dump(serializable_properties, prop_file, indent=4)

        return out_filename,prop_filename,num_pages

    except FileNotFoundError:
        return {"error": f"File '{pdf_path}' not found."}
    except Exception as e:
        return {"error": str(e)}

def clean_string(text):
    """
    Cleans a given text string by removing headers, speaker names, and unnecessary whitespace.

    This function is useful for processing transcripts, reports, or any text where:
    - Headers (assumed to be in all caps at the start of lines) should be removed.
    - Speaker names (assumed to be in all caps followed by a colon) should be removed.
    - Extra whitespace should be collapsed into a single space.
    - Sentences should be properly tokenized for improved readability.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text with headers, speaker names, and excessive whitespace removed.
        dict: Returns an error message in case of an exception.

    Raises:
        Exception: Catches any unexpected errors and returns an error message.

    Example:
        >>> raw_text = INTRODUCTION\nSPEAKER: Hello, everyone. \nThis is a sample text.\n\n\nNEXT HEADER\n JOHN: How are you?
        >>> clean_string(raw_text)
        "Hello, everyone. This is a sample text. How are you?"
    """
    try:
        # Remove headers (assuming headers are in all caps and appear at the start of a line)
        text = re.sub(r'^[A-Z\s]+$', '', text, flags=re.MULTILINE)

        # Remove speaker names (assuming speaker names are in all caps followed by a colon)
        text = re.sub(r'^[A-Z]+:', '', text, flags=re.MULTILINE)

        # Remove extra whitespace and ensure single spaces between sentences
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize into sentences for better readability
        sentences = sent_tokenize(text)
        cleaned_text = ' '.join(sentences)

        return cleaned_text

    except Exception as e:
        return {"error": str(e)}




def main():

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python texter.py <pdf_path> [output_path]")
        sys.exit(1)

    text_only = False
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    text_only_str = sys.argv[3] if len(sys.argv) >=4  else None
    if text_only_str == "text_only":
        text_only = True
    start_time = time.time()
    result = texter(pdf_path, output_path,text_only)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    if (len(result) >= 3):
        output_file = result[0]
        prop_file = result[1]
        page_count = result[2]
        print(f"\nElapsed time {round(elapsed_time)} seconds.\nText extracted from {page_count} pages to: {output_file}, PDF file properties in {prop_file} ")
    else:
        print(f"Failed: {result}")

if __name__ == "__main__":
    main()