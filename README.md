# Clausewise

A Streamlit legal document assistant orchestrated with LangGraph, ChromaDB, and OpenRouter.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add a rotated OpenRouter key to .env
streamlit run app.py
```

Upload a PDF, TXT, or Markdown contract. The app chunks it into a local Chroma collection, retrieves relevant passages for the question, then runs a LangGraph chain: retrieve -> extract -> analyze -> assess risk -> explain -> notify.

The UI supports a demo document when no upload is present, so the workflow can be inspected without an API key. Live answers require `OPENROUTER_API_KEY`.
# ai-legal-assitant
