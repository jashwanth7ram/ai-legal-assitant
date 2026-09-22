from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Protocol, TypedDict

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

APP_DIR = Path(__file__).parent
CHROMA_DIR = APP_DIR / ".chroma"
DEMO_TEXT = """SERVICE AGREEMENT\n\n1. SCOPE. Provider will deliver the services described in each statement of work.\n\n2. FEES AND PAYMENT. Client will pay undisputed invoices within thirty days. Late amounts accrue interest at 1.5% per month.\n\n3. TERM AND TERMINATION. This agreement begins on the effective date and continues for one year. Either party may terminate for convenience on thirty days written notice. Either party may terminate immediately for a material breach that remains uncured for fifteen days.\n\n4. LIMITATION OF LIABILITY. Neither party will be liable for indirect, incidental, special, or consequential damages. Each party's aggregate liability is limited to fees paid in the six months before the event giving rise to the claim.\n\n5. CONFIDENTIALITY. Each party will protect the other's confidential information using reasonable care and may use it only to perform this agreement.\n\n6. GOVERNING LAW. This agreement is governed by the laws of the State of New York, without regard to conflict of law principles.\n"""


class LegalState(TypedDict, total=False):
    question: str
    document_name: str
    document_text: str
    strategy: str
    retrieved: list[dict[str, Any]]
    clauses: list[dict[str, str]]
    risk_items: list[dict[str, str]]
    analysis: str
    answer: str
    citations: list[str]
    status: str
    error: str


class AnalysisStrategy(Protocol):
    name: str

    def instruction(self) -> str: ...


class RiskFirstStrategy:
    name = "Risk-first review"

    def instruction(self) -> str:
        return "Prioritize obligations, liability exposure, termination rights, and ambiguous language."


class PlainLanguageStrategy:
    name = "Plain-language summary"

    def instruction(self) -> str:
        return "Translate the relevant legal language into a concise explanation for a non-lawyer."


class ClauseComparisonStrategy:
    name = "Clause comparison"

    def instruction(self) -> str:
        return "Compare the relevant clauses by scope, duties, timing, exceptions, and practical impact."


class AnalyzerFactory:
    _strategies = {
        RiskFirstStrategy.name: RiskFirstStrategy,
        PlainLanguageStrategy.name: PlainLanguageStrategy,
        ClauseComparisonStrategy.name: ClauseComparisonStrategy,
    }

    @classmethod
    def create(cls, name: str) -> AnalysisStrategy:
        return cls._strategies.get(name, RiskFirstStrategy)()


class OpenRouterAdapter:
    """Adapter that keeps the graph independent from the OpenRouter SDK."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def complete(self, system: str, prompt: str) -> str:
        if not self.api_key or self.api_key.startswith("replace-"):
            return ""
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-vl:free"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class AnalysisObserver(Protocol):
    def update(self, message: str) -> None: ...


class SessionNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def update(self, message: str) -> None:
        self.messages.append(message)


def extract_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return DEMO_TEXT
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 850, overlap: int = 120) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + 250:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def get_collection(text: str, name: str):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    document_key = f"{name}\0{text}"
    collection_name = "contract_" + hashlib.sha1(document_key.encode("utf-8")).hexdigest()[:14]
    collection = client.get_or_create_collection(collection_name)
    if collection.count() == 0:
        chunks = chunk_text(text)
        collection.add(
            ids=[f"chunk-{index}" for index in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": name, "chunk": index + 1} for index in range(len(chunks))],
        )
    return collection


def retrieve_node(state: LegalState) -> LegalState:
    collection = get_collection(state["document_text"], state["document_name"])
    result = collection.query(query_texts=[state["question"]], n_results=min(5, collection.count()))
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    retrieved = [
        {"text": document, "source": metadata.get("source", state["document_name"]), "chunk": metadata.get("chunk", 0)}
        for document, metadata in zip(documents, metadatas)
    ]
    return {"retrieved": retrieved, "status": "Retrieved relevant passages"}


MODEL = OpenRouterAdapter()


def clean_model_markdown(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    cleaned = cleaned.replace("\\r", "").replace("\\n", "\n").replace('\\"', '"')
    return re.sub(r"\\([\\`*_{}\[\]()#+.!><-])", r"\1", cleaned)


def parse_model_json(raw: str) -> list[dict[str, str]]:
    import json

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
    cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def clause_tool_node(state: LegalState) -> LegalState:
    context = "\n\n".join(item["text"] for item in state["retrieved"])
    raw = MODEL.complete(
        "You extract legal clauses. Return JSON array only, with objects containing type, excerpt, and interpretation.",
        f"Question: {state['question']}\nContext:\n{context}",
    )
    clauses = []
    if raw:
        clauses = parse_model_json(raw)
        if not clauses:
            clauses = [{"type": "Relevant clause", "excerpt": clean_model_markdown(raw)[:400], "interpretation": "Model returned an unstructured extraction."}]
    if not clauses:
        keywords = [("termination", "Termination"), ("liability", "Limitation of liability"), ("payment", "Payment"), ("confidential", "Confidentiality")]
        clauses = [{"type": label, "excerpt": item["text"][:260], "interpretation": "Relevant passage retrieved from the uploaded document."} for item in state["retrieved"] for word, label in keywords if word in item["text"].lower()][:4]
    return {"clauses": clauses, "status": "Extracted relevant clauses"}


def risk_tool_node(state: LegalState) -> LegalState:
    context = "\n".join(f"- {item['text']}" for item in state["retrieved"])
    raw = MODEL.complete(
        "You are a cautious legal risk analyst. Return JSON array only, with severity, title, and detail. Do not give a definitive legal opinion.",
        f"Identify practical review risks for this question: {state['question']}\nRelevant text:\n{context}",
    )
    risks = []
    if raw:
        risks = parse_model_json(raw)
        if not risks:
            risks = [{"severity": "Review", "title": "Model assessment", "detail": clean_model_markdown(raw)[:500]}]
    if not risks:
        risks = [{"severity": "Medium", "title": "Review the liability cap", "detail": "Confirm that the six-month fee cap and damages exclusion match the commercial risk you intend to accept."}]
        if any("termination" in item["text"].lower() for item in state["retrieved"]):
            risks.append({"severity": "Medium", "title": "Termination timing", "detail": "Check whether thirty days for convenience and fifteen days to cure breach provide enough operational runway."})
    return {"risk_items": risks, "status": "Assessed review risks"}


def analysis_node(state: LegalState) -> LegalState:
    context = "\n\n".join(f"[Chunk {item['chunk']}] {item['text']}" for item in state["retrieved"])
    strategy = AnalyzerFactory.create(state["strategy"])
    answer = MODEL.complete(
        "You are a precise legal document assistant. Explain in plain language, distinguish document text from inference, and cite chunk numbers. Never claim to be a lawyer.",
        f"Strategy: {strategy.name}\nStrategy instruction: {strategy.instruction()}\nQuestion: {state['question']}\nDocument context:\n{context}",
    )
    if answer:
        answer = clean_model_markdown(answer)
    else:
        answer = "The document appears to address this through its payment, termination, and liability provisions. Review the extracted clauses below against your specific deal context. This demo response is intentionally conservative because no OpenRouter key is configured."
    citations = [f"{item['source']} · chunk {item['chunk']}" for item in state["retrieved"]]
    return {"analysis": answer, "answer": answer, "citations": citations, "status": "Prepared answer and citations"}


def notify_node(state: LegalState) -> LegalState:
    observer = SessionNotifier()
    observer.update(f"Analysis complete for {state['document_name']}")
    return {"status": observer.messages[-1]}


def build_graph():
    graph = StateGraph(LegalState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("extract_clauses", clause_tool_node)
    graph.add_node("assess_risk", risk_tool_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("notify", notify_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract_clauses")
    graph.add_edge("extract_clauses", "assess_risk")
    graph.add_edge("assess_risk", "analyze")
    graph.add_edge("analyze", "notify")
    graph.add_edge("notify", END)
    return graph.compile()


st.set_page_config(page_title="Clausewise", page_icon="§", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#ffffff; --muted:#ffffff; --line:#4b5563; --accent:#7dd3c0; --orange:#f5a36f; }
.stApp { color:var(--ink); font-family:'DM Sans', sans-serif; }
h1,h2,h3,h4,h5,h6,p,span,label,small { color:#ffffff; }
[data-testid="stCaptionContainer"] { color:#ffffff !important; }
h1,h2,h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:0; }
.block-container { max-width:1200px; padding-top:2.5rem; }
.hero { border-bottom:1px solid var(--line); padding-bottom:1.6rem; margin-bottom:1.35rem; }
.eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.15em; font-size:.7rem; font-weight:700; }
.hero h1 { font-size:3rem; margin:.35rem 0 .4rem; }
.hero p { color:var(--muted); max-width:650px; font-size:1.05rem; }
.workspace { padding:1.35rem; margin-bottom:1.5rem; }
.workspace h2 { margin:0 0 .25rem; font-size:1.35rem; }
.panel { padding:1.2rem; margin-bottom:1rem; }
.panel h3 { margin-top:0; font-size:1rem; }
.status { background:#e7f4ee; border-left:4px solid var(--accent); padding:.85rem 1rem; border-radius:4px; }
.status, .status * { color:#172126 !important; }
.clause { border-top:1px solid var(--line); padding:.85rem 0; }
.clause:first-child { border-top:0; padding-top:0; }
.pill { display:inline-block; border:1px solid #c8ded6; color:var(--accent); padding:.2rem .5rem; border-radius:999px; font-size:.72rem; font-weight:700; }
.risk { border-left:3px solid var(--orange); padding:.7rem .8rem; margin:.55rem 0; background:#fff8f1; }
.risk, .risk * { color:#172126 !important; }
.small { color:var(--muted); font-size:.82rem; }
.stFileUploader { margin-bottom:.35rem; }
.stButton > button { min-height:2.65rem; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">Private legal intelligence workspace</div><h1>Clausewise</h1><p>Ask a contract question. Trace the answer from retrieved document passages to clause interpretation and practical risk review.</p></div>', unsafe_allow_html=True)

st.markdown('<section class="workspace"><h2>Upload a contract</h2></section>', unsafe_allow_html=True)
uploaded = st.file_uploader("Choose a contract", type=["pdf", "txt", "md"], label_visibility="collapsed")
analyze = st.button("Submit", type="primary", use_container_width=True)
question = "What should I review before signing this agreement?"
strategy = "Risk-first review"

if "result" not in st.session_state:
    st.session_state.result = None
if analyze:
    if not uploaded:
        st.warning("Upload a document before submitting.")
    elif not MODEL.api_key or MODEL.api_key.startswith("replace-"):
        st.error("Add OPENROUTER_API_KEY to .env before submitting a document.")
    elif not question.strip():
        st.warning("Add a question before running the analysis.")
    else:
        document_name = uploaded.name if uploaded else "Demo service agreement"
        document_text = extract_text(uploaded)
        if not document_text.strip():
            st.error("No readable text was found in this PDF. Upload a text-based PDF or add OCR first.")
        else:
            with st.spinner("Analyzing the uploaded document..."):
                try:
                    st.session_state.result = build_graph().invoke({"question": question, "strategy": strategy, "document_name": document_name, "document_text": document_text})
                except Exception as exc:
                    st.error(f"Analysis could not complete: {exc}")

result = st.session_state.result
if result:
    st.markdown(f'<div class="status"><strong>{result.get("status", "Analysis complete")}</strong> · {result.get("document_name", "Document")} · {strategy}</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("### Answer")
        with st.container(border=True):
            st.markdown(result.get("answer", ""))
    with right:
        st.markdown("### Risk review")
        for risk in result.get("risk_items", []):
            st.markdown(f'<div class="risk"><strong>{risk.get("severity", "Review")} · {risk.get("title", "Item")}</strong><br><span class="small">{risk.get("detail", "")}</span></div>', unsafe_allow_html=True)
        st.markdown("### Clause tool")
        for clause in result.get("clauses", []):
            st.markdown(f'<div class="clause"><strong>{clause.get("type", "Clause")}</strong><br><span class="small">{clause.get("excerpt", "")}</span><br>{clause.get("interpretation", "")}</div>', unsafe_allow_html=True)
        st.markdown("### Citations")
        for citation in result.get("citations", []):
            st.markdown(f"- {citation}")
else:
    st.write("")
