python .\texter.py %1.pdf
python .\chunker.py %1.json
python .\summarizer.py %1-chunked.json
python .\pineconer.py %1-summarized.json %2 %3