"""
agent.py
The tool-using LangChain agent (TutoringInsightsAgent) that turns retrieved transcript
context into structured, actionable insights and can export them as a PDF report.
"""
import os
import json
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag_engine import RAGSearch

load_dotenv()


class TutoringInsights(BaseModel):
    topics: List[str] = Field(default_factory=list)
    student_understanding: Literal["low", "medium", "high"] = "medium"
    misconceptions: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)
    suggested_activities: List[str] = Field(default_factory=list)
    next_best_prompts: List[str] = Field(default_factory=list)
    actionable_feedback: List[str] = Field(default_factory=list)
    summary: str = ""
    difficulty_rating: int = Field(3, ge=1, le=5)
    blooms_level: Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"] = "Understand"
    sentiment: Literal["frustrated", "neutral", "confident", "excited", "confused"] = "neutral"
    key_quotes: List[str] = Field(default_factory=list)
    confidence: float = Field(0.6, ge=0, le=1)


DEFAULT_SYSTEM_PROMPT = (
    "You are a tutoring analytics coach. Read student-tutor transcripts and produce precise, "
    "actionable insights to plan the next step. Be diagnostic, concise, and concrete. "
    "Identify misconceptions and gaps; propose targeted questions and activities. "
    "Always return output that follows the requested format."
)

_rag: Optional[RAGSearch] = None


def get_rag() -> RAGSearch:
    """Lazily initializes the shared RAGSearch instance (so importing this module doesn't
    require API keys / a built index until an insight is actually requested)."""
    global _rag
    if _rag is None:
        _rag = RAGSearch(persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2",
                          llm_model="llama-3.1-8b-instant")
    return _rag


@tool("rag_search", return_direct=False)
def rag_search(query: str) -> str:
    """Retrieve top-k transcript chunks for the user's query from the FAISS-backed vector store."""
    results = get_rag().vectorstore.query(query, top_k=5)
    texts = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata")]
    joined = "\n\n".join(t for t in texts if t)
    return joined if joined.strip() else "[EMPTY]"


def _save_tutoring_insights_pdf_core(insights: Dict[str, Any], retrieved_chunks: Optional[List[str]] = None,
                                      output_path: str = "tutoring_insights.pdf") -> str:
    retrieved_chunks = retrieved_chunks or []
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Tutoring Session Insights Report</b>", styles["Title"]),
        Paragraph(datetime.now().strftime("%B %d, %Y %H:%M"), styles["Normal"]),
        Spacer(1, 12),
        Paragraph("<b>Summary</b>", styles["Heading2"]),
        Paragraph(insights.get("summary", "No summary."), styles["BodyText"]),
        Spacer(1, 12),
    ]

    core_data = [
        ["Student Understanding", insights.get("student_understanding", "")],
        ["Difficulty Rating", str(insights.get("difficulty_rating", ""))],
        ["Bloom's Level", insights.get("blooms_level", "")],
        ["Sentiment", insights.get("sentiment", "")],
        ["Confidence", f"{float(insights.get('confidence', 0.0)):.2f}"],
    ]
    table = Table(core_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    story += [Paragraph("<b>Session Metrics</b>", styles["Heading2"]), table, Spacer(1, 12)]

    for field in ["topics", "misconceptions", "knowledge_gaps", "questions_to_ask",
                  "suggested_activities", "next_best_prompts", "actionable_feedback", "key_quotes"]:
        vals = insights.get(field, [])
        if vals:
            story.append(Paragraph(f"<b>{field.replace('_', ' ').title()}</b>", styles["Heading3"]))
            for item in vals:
                story.append(Paragraph(f"- {item}", styles["BodyText"]))
            story.append(Spacer(1, 6))

    if retrieved_chunks:
        story += [Spacer(1, 12), Paragraph("<b>Retrieved Transcript Chunks</b>", styles["Heading2"])]
        for i, chunk in enumerate(retrieved_chunks[:5], 1):
            snippet = chunk[:500] + ("..." if len(chunk) > 500 else "")
            story += [Paragraph(f"[{i}] {snippet}", styles["Code"]), Spacer(1, 4)]

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out_path), pagesize=A4).build(story)
    return str(out_path)


@tool("save_tutoring_insights_pdf", return_direct=False)
def save_tutoring_insights_pdf(payload: str) -> str:
    """Save TutoringInsights to a PDF. Payload is a JSON string with keys:
    insights (dict, required), retrieved_chunks (list[str], optional), output_path (str, optional)."""
    try:
        data = json.loads(payload)
        insights = data.get("insights")
        if not isinstance(insights, dict):
            return "[ERROR] 'insights' must be a JSON object."
        return f"[INFO] PDF saved: {_save_tutoring_insights_pdf_core(insights, data.get('retrieved_chunks', []), data.get('output_path', 'tutoring_insights.pdf'))}"
    except json.JSONDecodeError:
        return "[ERROR] Payload must be valid JSON."
    except Exception as e:
        return f"[ERROR] Failed to save PDF: {e}"


def build_agent_executor() -> AgentExecutor:
    """Builds the OpenAI tools agent: it calls rag_search to ground itself in real transcript
    context, then optionally calls save_tutoring_insights_pdf to export a report."""
    parser = PydanticOutputParser(pydantic_object=TutoringInsights)
    agent_system_prompt = (
        DEFAULT_SYSTEM_PROMPT
        + "\n\nProcess:\n"
        "1) First call the `rag_search` tool with the user's query to fetch transcript excerpts.\n"
        "2) Use those excerpts to produce insights that strictly match the requested format.\n"
        "3) If the user asks to save as PDF, call `save_tutoring_insights_pdf` with a JSON payload "
        "containing the insights, retrieved_chunks (optional), and output_path (optional).\n"
        "Return only the formatted insights unless explicitly asked to save."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_system_prompt + "\n{format_instructions}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                      openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    tools = [rag_search, save_tutoring_insights_pdf]
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
