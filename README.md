# Legal Document Assistant

An Agentic AI capstone project that helps paralegals and junior lawyers ask grounded questions over legal-document review knowledge.

## Student Details

- Name: Ria Agrawal
- Roll Number: 23053069
- Batch/Program: CSE B.Tech

## Problem Statement

Legal teams spend significant time reading contracts, case notes, filing checklists, and compliance documents. A paralegal or junior lawyer often needs fast answers, but generic chatbots can hallucinate legal details. This project builds a legal document assistant that retrieves from a controlled legal knowledge base, answers only from available context, remembers conversation details during a session, uses a utility tool when needed, and admits uncertainty instead of inventing information.

## Key Features

- RAG-based answers from exactly 10 legal knowledge-base documents.
- LangGraph-style workflow with 8 nodes: memory, router, retrieval, skip retrieval, tool, answer, eval, and save.
- Session memory using `thread_id`.
- Tool route for date, deadline, and simple calculation questions.
- Faithfulness evaluator with retry logic.
- Streamlit browser interface.
- Red-team handling for prompt injection and legal-outcome prediction questions.
- Baseline evaluation script with RAGAS-style fallback scoring.

## Tech Stack

- Python
- Streamlit
- LangGraph
- ChromaDB
- SentenceTransformers
- RAGAS-style evaluation
- Pytest

## Project Structure

```text
legal-document-assistant/
  agent.py
  capstone_streamlit.py
  day13_capstone.ipynb
  requirements.txt
  legal_assistant/
    kb.py
    state.py
    retrieval.py
    tools.py
    nodes.py
    graph.py
    evaluation.py
  tests/
    test_agent.py
  docs/
    Project_Documentation.pdf
    screenshots/
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the command-line smoke demo:

```bash
python agent.py
```

4. Launch the UI:

```bash
streamlit run capstone_streamlit.py
```

The project includes lightweight fallbacks for local demonstration if ChromaDB, SentenceTransformers, or LangGraph are not installed. Install the requirements for the full capstone toolchain.

## Evaluation

The project includes tests for retrieval quality, memory, tool routing, prompt-injection refusal, and out-of-scope legal advice boundaries. It also includes a baseline evaluation module with five ground-truth question-answer pairs.

## Documentation

The complete project report is available at:

```text
docs/Project_Documentation.pdf
```
