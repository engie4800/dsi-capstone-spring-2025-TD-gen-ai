import time
import re
import json
import nltk
import spacy
import sys
import os
from dateutil.parser import parse
from collections import Counter


# Ensure you have the necessary NLTK data files and spaCy model
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')


def classifier(json_filename, properties_filename=None, num_chars=100000):
    """
    Classifies a document based on its content from a JSON file (output from pdf_to_text).
    Optionally extracts metadata from a separate JSON file.

    Parameters:
    json_filename (str): The path to the JSON file containing extracted PDF pages.
    metadata_filename (str, optional): The path to a JSON file containing metadata.
                                       Defaults to None.
    num_chars (int, optional): The number of characters to use for classification.
                               Defaults to 100000.

    Returns:
    str: A JSON string containing the document type, category, confidence,
         creation date, and modification date.

    Example:
    classify_text('sample.json', 'sample-prop.json')
    """
    try:

        # default the pdf prop filename
        if properties_filename is None:
            base_name = os.path.splitext(json_filename)[0]
            properties_filename = f"{base_name}-prop.json"

        print(f"Classifying {json_filename} using file contents and property file {properties_filename}.")

        # Load text from the JSON file
        with open(json_filename, 'r', encoding='utf-8') as json_file:
            pages = json.load(json_file)

        if not isinstance(pages, list) or not all("page_content" in p for p in pages):
            return json.dumps({"error": f"Invalid JSON format in '{json_filename}'."}, indent=4)

        # Combine all pages into a single text string (limited to num_chars)
        text = " ".join(page["page_content"] for page in pages)[:num_chars]

    except FileNotFoundError:
        return json.dumps({"error": f"File '{json_filename}' not found."}, indent=4)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Error decoding JSON from '{json_filename}'."}, indent=4)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

    # Define keywords for types and categories
    type_keywords = {
        "Financial": ["financial", "earnings", "statement", "report", "quarterly"],
        "Regulatory": ["regulatory", "compliance", "regulation", "law", "policy"],
        "Marketing": ["marketing", "advertisement", "campaign", "promotion", "brand"]
    }

    category_keywords = {
        "annual report": ["annual report", "yearly report"],
        "earnings call transcript": ["earnings call", "transcript"],
        "financial statement": ["financial statement", "balance sheet", "income statement"],
        "quarterly report": ["quarterly report", "q1", "q2", "q3", "q4"]
    }

    # Determine document type
    document_type, type_confidence = determine_type_or_category(json_filename, text, type_keywords)

    # Determine document category
    document_category, category_confidence = determine_type_or_category(json_filename, text, category_keywords)

    # Extract metadata (creation date & modification date)
    creationdate = "Unknown"
    moddate = "Unknown"

    if properties_filename:
        try:
            with open(properties_filename, 'r', encoding='utf-8') as metadata_file:
                metadata = json.load(metadata_file)
                creationdate = metadata.get("creationdate", "Unknown")
                moddate = metadata.get("moddate", "Unknown")
                source = metadata.get("source", "Unknown")
                filename = os.path.basename(source)
        except FileNotFoundError:
            return json.dumps({"error": f"Properties file '{properties_filename}' not found."}, indent=4)
        except json.JSONDecodeError:
            return json.dumps({"error": f"Error decoding JSON from properties file '{properties_filename}'."}, indent=4)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=4)

    # Determine overall confidence
    confidence = determine_confidence(type_confidence, category_confidence)

    # Create result JSON
    result = {
        # this adjustment seems to match old code
        #"document_type": document_type,
        "document_type": document_category,

        "publication_date": creationdate,
        "file_name": filename,
        #"document_category": document_category,
        # "confidence": confidence,
        # "document_type": document_type,
        # "creation_date": creationdate,
        # "modification_date": moddate,
        # "document_category": document_category,
        # "confidence": confidence,
        # "file_name" : filename
    }

    return json.dumps(result, indent=4)



def determine_type_or_category(filename, text, keywords_dict):
    # Combine filename and text for analysis
    combined_text = filename.lower() + " " + text.lower()

    # Count keyword occurrences
    keyword_counts = Counter()
    for key, keywords in keywords_dict.items():
        for keyword in keywords:
            keyword_counts[key] += combined_text.count(keyword)

    # Determine the most likely type or category
    if keyword_counts:
        most_common = keyword_counts.most_common(1)[0]
        if most_common[1] > 0:
            confidence = "high" if most_common[1] > 5 else "medium" if most_common[1] > 2 else "low"
            return most_common[0], confidence
    return "Unknown", "low"


def determine_publication_date(text):
    # Regular expressions to find various date formats
    date_patterns = [
        r'\b(\d{4})-(\d{2})-(\d{2})\b',  # YYYY-MM-DD
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  # MM/DD/YYYY
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  # MM-DD-YYYY
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b'
        # DD Month YYYY
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                date = parse(match.group())
                return date.strftime('%Y-%m-%d')
            except ValueError:
                continue
    return "Unknown"


def determine_confidence(type_confidence, category_confidence):
    if type_confidence == "high" and category_confidence == "high":
        return "high"
    elif type_confidence == "low" or category_confidence == "low":
        return "low"
    else:
        return "medium"


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python classifier.py <cleaned_input_filename> <metadata_filename> [num_chars]")
        sys.exit(1)

    input_filename = sys.argv[1]
    metadata_filename = sys.argv[2] if len(sys.argv) > 3 else None
    num_chars = int(sys.argv[3]) if len(sys.argv) > 4 else 100000
    start_time = time.time()
    result = classifier(input_filename,metadata_filename, num_chars)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"\nElapsed time {round(elapsed_time)} seconds.\n{result}")


if __name__ == "__main__":
    main()