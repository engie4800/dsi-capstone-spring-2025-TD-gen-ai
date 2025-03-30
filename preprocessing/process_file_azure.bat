python .\texter.py %1.pdf
python .\chunker.py %1.json
python .\summarizer.py %1-chunked.json %4 %5 --llm_model_name %6
python .\pineconer.py %1-summarized.json %2 %3 %4 %5 --embedding_model_name %9 %7 %8 