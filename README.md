# Legal Document Assistant

Agentic AI capstone project for a paralegal or junior lawyer who needs quick, grounded answers from legal documents.

## Student Details

- Name: Ria Agrawal
- Roll Number: 23053069
- Batch/Program: CSE B.Tech

## Problem Statement

Legal teams spend significant time reading contracts, case notes, filing checklists, and compliance documents. A paralegal or junior lawyer often needs fast answers, but generic chatbots can hallucinate legal details. This project builds a legal document assistant that retrieves from a controlled legal knowledge base, answers only from available context, remembers conversation details during a session, uses a utility tool when needed, and admits uncertainty instead of inventing information.

## Capstone Requirements Covered

- LangGraph-style `StateGraph` workflow with 8 nodes: memory, router, retrieval, skip retrieval, tool, answer, eval, save.
- ChromaDB RAG over exactly 10 legal knowledge documents.
- SentenceTransformer `all-MiniLM-L6-v2` embeddings when installed.
- MemorySaver with `thread_id` when LangGraph is installed.
- Tool route for date, deadline, and simple legal fee or page-count calculations.
- Faithfulness evaluator with retry gate.
- Streamlit deployment in `capstone_streamlit.py`.
- Baseline evaluation script with RAGAS-style fallback scoring.
- Tests including memory, out-of-scope, and prompt-injection red-team cases.

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
  submission/
    Legal_Document_Assistant_Submission.zip
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

## Submission

Submit the public GitHub repository link, the project ZIP file, and `docs/Project_Documentation.pdf` in the capstone Google Form.
