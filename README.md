# Azure OpenAI Document Processing & Semantic Search

A working demonstration of the AI architecture I shipped to production at BNY Mellon:
AI-powered document processing and semantic search built on Azure OpenAI.

The production system is internal to the bank, so this public version re-creates the
same architecture end to end with the same shapes, running locally with no API key,
so anyone can run it and see exactly how it works.

## What it does

1. **Load and chunk documents** — source documents are split into small, searchable chunks.
2. **Embeddings** — each chunk is turned into a vector representation of its meaning.
3. **Semantic search** — a question is embedded the same way, and the chunks whose
   meaning is closest are retrieved.
4. **Answer from sources** — the answer is generated from the retrieved chunks only,
   so it stays grounded in the documents.
5. **Validation** — predefined test queries are checked against the source documents:
   expected fact present, answer grounded in sources. All tests pass (see sample-output.txt).

In production, steps 2–4 call the Azure OpenAI APIs. Output quality is validated with
predefined test queries, comparison of generated answers against source documents,
human review, relevance and accuracy checks, edge-case testing, and monitoring of
responses for hallucinations.

## Run it

```
python3 azure_openai_doc_search.py
```

No dependencies beyond the Python standard library.

## Author

Abel Haile Magicho — Senior Software / AI Engineer
AI & Distributed Systems Engineer, BNY Mellon (Houston)
M.Sc. Software Engineering, Chalmers University of Technology
