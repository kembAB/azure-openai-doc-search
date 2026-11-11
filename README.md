# Azure OpenAI Document Processing & Semantic Search

A working demonstration of the AI architecture I shipped to production at BNY Mellon:
AI-powered document processing and semantic search built on Azure OpenAI — extended here
with an agentic layer, an automated eval suite, and a REST API, the three shapes most of
my current AI work takes.

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

## The files

- `azure_openai_doc_search.py` — the core pipeline above. Run: `python3 azure_openai_doc_search.py`
- `agent.py` — an agentic layer over the pipeline. The agent plans, calls tools
  (search, answer, grounding check), verifies its own draft, widens the search and
  retries once on failure — and refuses honestly when the documents cannot answer,
  instead of inventing. Run: `python3 agent.py`
- `evals.py` — an automated eval suite for the agent: 11 cases with known answers,
  paraphrase variants, and two questions the documents cannot answer, where the
  correct behaviour is refusal. Run: `python3 evals.py`
- `api.py` — a small REST API over the agent (standard library only):
  `GET /health` and `GET /ask?q=...`, answering with JSON that carries the answer,
  its sources, and its grounding status. Run: `python3 api.py`
- `sample-output.txt` — real output of the core program.
- `agent-evals-output.txt` — real output of the agent demo, the eval suite, and live API tests.

## The eval story

The first eval run scored 8 of 11: the refusal cases failed because small function
words made every question match something, and a paraphrase ("sick days") lost to
word repetition. The fixes were retrieval-side, not test-side: matching now ignores
stopwords and lightly stems words, sentence scoring weights rare shared words more,
and the agent gates answers on source coverage. The suite now passes 11 of 11,
and the core program's own validation still passes 3 of 3. The tests were not edited
to pass; the system was.

## Run it

```
python3 azure_openai_doc_search.py   # core pipeline + validation
python3 agent.py                     # agentic layer demo
python3 evals.py                     # eval suite (11 of 11)
python3 api.py                       # REST API on http://127.0.0.1:8765
```

No dependencies beyond the Python standard library.

## Author

Abel Haile Magicho — Senior Software / AI Engineer
AI & Distributed Systems Engineer, BNY Mellon (Houston)
M.Sc. Software Engineering, Chalmers University of Technology
