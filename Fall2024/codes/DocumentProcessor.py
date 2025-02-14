from pathlib import Path
import json
import uuid
from datetime import datetime
from time import sleep
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


class DocumentProcessor:
    def __init__(self, secrets: str):
        """Initialize DocumentProcessor with secrets from a JSON file.

        Args:
            secrets_path (str): Path to the secrets JSON file containing API keys
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=secrets["pinecone_api_key"])
        self.serverless_spec = ServerlessSpec(
            cloud='gcp',
            region='starter'
        )
        self.index = self.pc.Index("td-bank-docs", spec=self.serverless_spec)

        # Initialize other components
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = OpenAIEmbeddings()

    def find_pdf_files(self, root_dir: str) -> List[Path]:
        """Recursively find all PDF files in the root directory"""
        root_path = Path(root_dir)
        pdf_files = list(root_path.rglob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files in {root_dir}")
        return pdf_files

    def extract_document_info(self, file_path: str, content_preview: str) -> dict:
        """Extract document metadata including type and publication date"""
        try:
            file_name = Path(file_path).name

            prompt = f"""
            Based on the filename '{file_name}' and the following content preview, determine:
            1. Document type (e.g., annual report, earnings call transcript, financial statement, etc.)
            2. Publication date or relevant date
            3. Document category (financial, regulatory, marketing, etc.)

            Respond in JSON format:
            {{
                "document_type": "type here",
                "publication_date": "YYYY-MM-DD",
                "document_category": "category here",
                "confidence": "high/medium/low"
            }}

            Content Preview:
            {content_preview[:1000]}
            """

            response = self.llm.invoke(prompt)
            doc_info = json.loads(response.content)

            doc_info.update({
                "file_name": file_name,
                "file_path": str(Path(file_path).absolute()),
                "processing_date": datetime.now().strftime("%Y-%m-%d")
            })

            return doc_info
        except Exception as e:
            print(f"Error extracting document info: {e}")
            return {
                "document_type": "unknown",
                "publication_date": "unknown",
                "document_category": "unknown",
                "confidence": "low",
                "file_name": Path(file_path).name,
                "file_path": str(Path(file_path).absolute()),
                "processing_date": datetime.now().strftime("%Y-%m-%d")
            }

    def generate_section_summary(self, content: str) -> Dict:
        """Generate a concise summary of the section content."""
        try:
            prompt = f"""
            Provide a concise summary of the following content. Include:
            1. Main topic/theme (1 sentence)
            2. Key points (2-3 bullet points)
            3. Type of content (e.g., financial data, narrative, analysis)

            Content: {content[:3000]}...

            Respond in JSON format:
            {{
                "main_theme": "theme here",
                "key_points": ["point1", "point2", "point3"],
                "content_type": "type here",
                "brief_summary": "2-3 sentence summary here"
            }}
            """

            response = self.llm.invoke(prompt)
            return json.loads(response.content)
        except json.JSONDecodeError:
            print(f"\n⚠️ JSON Decode Error in summary. Response content was:\n{response.content}")
            return {
                "main_theme": "Error processing theme",
                "key_points": [],
                "content_type": "unknown",
                "brief_summary": "Error generating summary"
            }
        except Exception as e:
            print(f"Error generating summary: {e}")
            return {
                "main_theme": "Error processing theme",
                "key_points": [],
                "content_type": "unknown",
                "brief_summary": "Error generating summary"
            }

    def clean_content_with_llm(self, content: str) -> str:
        """Use LLM to clean content by removing headers, footers, and formatting issues."""
        cleaning_prompt = f"""Please clean up the following text by removing headers, speaker names, and formatting issues.
        Retain all meaningful content and ensure it flows naturally for readability. The text should contain no unnecessary information.
        Text to clean: {content}"""

        response = self.llm.invoke(cleaning_prompt)
        cleaned_content = response.content.strip() if response and hasattr(response, 'content') else content
        return cleaned_content

    def process_pdf_by_sections(self, file_path: str) -> tuple:
        """Process PDF by detecting topic changes with a rolling window approach."""
        start_time = datetime.now()
        print(f"\n[Loading PDF] Processing: {Path(file_path).name}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        total_pages = len(pages)
        print(f"✓ Loaded PDF with {total_pages} pages")

        # Extract document information
        try:
            doc_info = self.extract_document_info(file_path, pages[0].page_content)
        except Exception as e:
            print(f"Error extracting document info: {e}")
            doc_info = {
                'file_name': Path(file_path).name,
                'document_type': 'Unknown',
                'publication_date': 'Unknown',
                'document_category': 'Unknown'
            }

        # Clean and prepare page text
        print("[Cleaning Pages] Using LLM to clean content from each page...")
        page_texts = []
        for page in tqdm(pages, desc="Cleaning Pages"):
            cleaned_text = self.clean_content_with_llm(page.page_content)
            page_texts.append(cleaned_text)
        print("✓ Completed cleaning all pages")

        # Generate embeddings
        print("[Embedding Pages] Generating embeddings for each page...")
        page_embeddings = self.sentence_transformer.encode(page_texts)
        print("✓ Completed embedding all pages")

        # Rolling window settings
        similarity_threshold = 0.7
        sections = []
        section_start_page = 0

        print("[Detecting Topics] Running topic change detection with rolling window...")
        for i in tqdm(range(1, len(page_embeddings)), desc="Detecting Topics"):
            sim_score = cosine_similarity([page_embeddings[i - 1]], [page_embeddings[i]])[0][0]

            if sim_score < similarity_threshold:
                section_text = "\n".join(page_texts[section_start_page:i])

                # Get topic and summary
                summary_data = self.generate_section_summary(section_text)

                page_range = str(
                    section_start_page + 1) if section_start_page + 1 == i else f"{section_start_page + 1}-{i}"

                sections.append({
                    'content': section_text,
                    'page_range': page_range,
                    'topic': summary_data['main_theme'],
                    'summary': summary_data['brief_summary']
                })

                section_start_page = i

        # Process final section
        if section_start_page < total_pages:
            section_text = "\n".join(page_texts[section_start_page:])
            summary_data = self.generate_section_summary(section_text)

            page_range = str(
                section_start_page + 1) if section_start_page + 1 == total_pages else f"{section_start_page + 1}-{total_pages}"

            sections.append({
                'content': section_text,
                'page_range': page_range,
                'topic': summary_data['main_theme'],
                'summary': summary_data['brief_summary']
            })

        print("✓ Completed topic detection and section creation")

        # Format document chunks with metadata
        processed_chunks = [
            Document(
                page_content=section['content'],
                metadata={
                    'page_range': section['page_range'],
                    'file_name': doc_info['file_name'],
                    'document_type': doc_info['document_type'],
                    'publication_date': doc_info['publication_date'],
                    'document_category': doc_info['document_category'],
                    'topic': section['topic'],
                    'summary': section['summary']
                }
            )
            for section in sections
        ]

        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✓ Completed processing in {processing_time:.1f} seconds")
        print(f"✓ Identified {len(processed_chunks)} topic-based sections")

        return processed_chunks, doc_info

    def upload_to_pinecone(self, chunks: List[Document], batch_size: int = 100) -> int:
        """Upload processed chunks to Pinecone with essential metadata."""
        try:
            pinecone_data = []

            print("\nGenerating embeddings...")
            for chunk in tqdm(chunks, desc="Generating embeddings"):
                try:
                    content_embedding = self.embeddings.embed_query(chunk.page_content)

                    metadata = {
                        "content": chunk.page_content[:40000],
                        "summary": chunk.metadata.get("summary", ""),
                        "document_type": chunk.metadata.get("document_type", ""),
                        "publication_date": chunk.metadata.get("publication_date", ""),
                        "file_name": chunk.metadata.get("file_name", ""),
                        "page_range": chunk.metadata.get("page_range", ""),
                        "content_type": chunk.metadata.get("content_type", "")
                    }

                    pinecone_data.append({
                        "id": f"content_{str(uuid.uuid4())}",
                        "values": content_embedding,
                        "metadata": metadata
                    })

                except Exception as e:
                    print(
                        f"\n⚠️ Error generating embeddings for chunk {chunk.metadata.get('section_number', 'unknown')}: {e}")
                    continue

            print("\nUploading to Pinecone...")
            for i in range(0, len(pinecone_data), batch_size):
                try:
                    batch = pinecone_data[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                    print(f"Uploaded batch {i // batch_size + 1}/{len(pinecone_data) // batch_size + 1}")
                    sleep(1)  # Rate limiting

                except Exception as e:
                    print(f"\n⚠️ Error uploading batch {i // batch_size + 1}: {e}")
                    sleep(60)  # Longer delay on error
                    continue

            print(f"\n✓ Successfully uploaded {len(pinecone_data)} vectors to Pinecone")
            return len(pinecone_data)

        except Exception as e:
            print(f"\n❌ Fatal error in upload_to_pinecone: {e}")
            raise e

    def process_all_pdfs(self, root_dir: str):
        """Process all PDFs in directory and upload to Pinecone

        Args:
            root_dir (str): Root directory containing PDF files
        """
        # Find all PDFs
        pdf_files = self.find_pdf_files(root_dir)

        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\nProcessing file {i}/{len(pdf_files)}: {pdf_path.name}")
            try:
                # Process PDF
                chunks, doc_info = self.process_pdf_by_sections(str(pdf_path))

                # Upload to Pinecone
                vectors_uploaded = self.upload_to_pinecone(chunks)

                print(f"Successfully processed {pdf_path.name}")
                print(f"Uploaded {vectors_uploaded} vectors")

            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue

            # Add delay between files
            sleep(2)

    def search_pinecone(self, query: str, top_k: int = 4) -> List[Dict]:
        """Search Pinecone index for similar documents

        Args:
            query (str): The search query
            top_k (int, optional): Number of results to return. Defaults to 4.

        Returns:
            List[Dict]: List of search results with metadata
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Print results
        print(f"\nTop {top_k} results for query: '{query}'")
        for i, result in enumerate(search_results['matches'], 1):
            metadata = result.get('metadata', {})
            document_type = metadata.get('document_type', 'Unknown Document Type')
            file_name = metadata.get('file_name', 'Unknown File Name')
            publication_date = metadata.get('publication_date', 'Unknown Date')
            page_range = metadata.get('page_range', 'Unknown Page Range')
            chunk_type = metadata.get('chunk_type', 'Unknown Content Type')
            content_preview = metadata.get('content', 'No content available')[:200]

            print(f"\nResult {i} (Score: {result['score']:.4f})")
            print(f"Document: {document_type} ({file_name})")
            print(f"Published: {publication_date}")
            print(f"Pages: {page_range}")
            print(f"Content Type: {chunk_type}")
            print(f"Content Preview: {content_preview}...")

        return search_results['matches']
