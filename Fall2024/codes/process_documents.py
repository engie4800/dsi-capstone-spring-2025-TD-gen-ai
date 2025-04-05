import json
from pathlib import Path
from DocumentProcessor import DocumentProcessor

def main():
    # Load secrets from secrets.json
    secrets_path = Path(__file__).parent.parent / "secrets.json"
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    
    # Initialize the document processor
    processor = DocumentProcessor(secrets)
    
    # Specify the directory containing PDF files
    pdf_directory = Path(__file__).parent.parent / "pdfs"  # Change this path to your PDF directory
    
    # Process all PDFs in the directory
    processor.process_all_pdfs(str(pdf_directory))

if __name__ == "__main__":
    main() 