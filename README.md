# Data Science Capstone & Ethics (ENGI E4800)

## Overview

TD Bank has partnered with Columbia University’s Master of Data Science program to develop a Generative AI Executive Advisor that helps executives quickly access strategic, evidence-based insights. Faced with reviewing large volumes of reports, proposals, and financial documents, executives need a tool that can cut through the noise and surface what matters most. Using Retrieval-Augmented Generation (RAG) techniques with GPT-4o and other large language models, the system will process public financial documents, such as annual reports, earnings call transcripts, and regulatory filings, and answer executive-level questions with speed and accuracy. The goal is to reduce the time and effort required to find key insights, ensuring that the information provided is reliable, relevant, and clearly supported by source data. Led by Daniel Randles and Mohammad Nejad from TD Bank, this AI-powered solution is designed to act as a virtual executive assistant, supporting faster, smarter, and more confident decision-making.

## Team Structure

1. Xiqian Yuan, xy2655 (team captain)
2. Sam Gabor, sg662
3. Gregor Zdunski Hanuschak, gzh2101
4. Brianna Hoang Yen Ta, bht2118
5. Dai Dai, yd2765

## Features
- Intelligent Q&A: Answer natural language questions based on the content of financial documents.
- Evidence-Based Responses: Provides references (document names and page numbers) for generated answers to ensure reliability and allow users to verify information.
- Advanced Retrieval: Implements Hybrid Search, combining semantic and keyword search for improved document relevance.
- Enhanced Ranking: Utilizes Reranking techniques (Cohere Rerank and Two-Stage Rerank) to refine search results and prioritize the most relevant information.
- Abstention Mechanism: Incorporates a confidence-based abstention system using G-Eval to mitigate hallucinations and withhold responses when confidence is low.
- Configurable Pipeline: Modular data processing pipeline allowing for independent optimization of steps.
- LLM Flexibility: Supports various LLMs and infrastructures (OpenAI, Ollama, Azure).
- Interactive UI: Chatbot interface with options to control technical parameters like Reranking, Confidence Threshold, and Hybrid Search weighting.

## Methods
The system is built around an RAG architecture, allowing the LLM to retrieve information from a specified knowledge base before generating a response.

### Data
The knowledge base consists of 27 publicly available PDF documents detailing TD Bank's financial information from 2021 to 2025. This includes quarterly financial reports, annual reports, and earnings call transcripts, downloaded from the official TD Bank website.

### Document Processing Pipeline
PDF documents are processed through a five-step pipeline before being indexed for retrieval:
- Conversion to text: PDFs are converted into JSON format, preserving page text and original page numbers.
- Related text chunking: Pages are grouped into logical chunks based on cosine similarity, potentially spanning multiple pages, to maintain context.
- Chunk summarization: An LLM generates a concise summary for each text chunk.
- Document classification: (Implicit in metadata based on source)
- Indexing with Metadata: Chunks, summaries, source page ranges, original document date ranges, document types, and vector embeddings (both dense and sparse) are uploaded to a Pinecone vector database.
Output from each stage is preserved for review and debugging.

### Retrieval Mechanism
- Hybrid Search: Combines dense vector embeddings (semantic similarity) with sparse vector embeddings (keyword matching) to retrieve an initial set of potentially relevant document chunks from the Pinecone database. The balance between semantic and keyword search is controlled by an adjustable alpha weighting value.
- Reranking: The initial retrieved chunks are then reordered to improve relevance using one of two methods:
Cohere Rerank: Leverages the Cohere rerank-english-v3.0 model to assign relevance scores based on nuanced contextual similarity.
- Two-Stage Rerank: Employs a fast retrieval embedding model followed by a computationally intensive cross-encoder reranking model to refine the order of results.
- Abstention: To prevent the generation of potentially hallucinated responses. It utilizes the G-Eval framework. 

### Performance
Significant performance improvements were achieved in the document processing pipeline. A complete load of the 27-document corpus, comprising thousands of pages, now takes approximately 2.5 hours, a tenfold improvement over the original system's 24+ hours. This enhancement facilitates quicker integration of new documents.

### Evaluation
The system's answer evaluation performance was assessed using a testbed of 160 questions (80 answerable, 80 unanswerable) derived from one of the documents. On this evaluation, the system achieved:
- Precision: 85%
- Recall: 89%
Precision measures the accuracy of the affirmative answers provided (proportion of correct answers among all answers given). Recall measures the system's ability to find all relevant answers (proportion of correct answers found among all answerable questions).

### User Interface
The chatbot interface includes several options to facilitate experimentation and parameter tuning:
- Trace On: Enables display of console messages for debugging.
- Use Cohere Reranker: Toggles the use of the Cohere API for reranking retrieved documents.
- Use Confidence Threshold: Allows setting a threshold (0-1) for the abstention mechanism.
- Use Hybrid Search: Allows specifying the alpha weighting value for the keyword-based component of the hybrid search.

### Future Exploration
Several areas have been identified for further development and improvement:
1. Expand Evaluation Framework: Create a larger, more diverse testbed of questions to enable comprehensive multi-variable analysis and systematic tuning of parameters for optimal precision, recall, and abstention.
2. Enhance Document Processing Pipeline: Experiment with state-of-the-art and potentially domain-specific embedding models, and refine chunking and summarization logic to improve initial retrieval quality.
3. Investigate Cloud Deployment: Explore a production-grade, scalable cloud infrastructure deployment (Azure, AWS) with containerization, secure APIs, and autoscaling to support live executive interaction and telemetry collection.
   
