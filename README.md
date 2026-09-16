# 🧑‍🏫 Teacher Intelligence Agent

A Retrieval-Augmented Generation (RAG) system that reads student-tutor transcripts and
turns them into structured, actionable insights — understanding level, misconceptions,
sentiment, suggested next steps — with an optional one-click PDF report for educators.

---

## What it does

Given a folder of tutoring session transcripts, the agent can answer questions like:

> *"Did the student understand the three states of matter? What should the tutor do next?"*

...and return either a quick semantic-search summary, or a full structured report covering:

- Topics covered & knowledge gaps
- Misconceptions identified
- Student sentiment & confidence
- Bloom's taxonomy level
- Suggested follow-up questions and activities
- A downloadable PDF report

## Architecture

```
Transcripts (JSON)
      │
      ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
      │
      ▼
Embeddings (SentenceTransformers: all-MiniLM-L6-v2)
      │
      ▼
FAISS vector index  ──────────────┐
      │                            │
      ▼                            ▼
RAGSearch (Groq Llama 3.1)   LangChain Agent (OpenAI gpt-4o-mini)
  quick summaries                tool-calling: rag_search → structured
                                  Pydantic insights → save_tutoring_insights_pdf
```

The agent is a real **tool-calling LangChain agent**, not a fixed pipeline: it decides
when to call `rag_search` to ground itself in transcript evidence, and when to call
`save_tutoring_insights_pdf` to export a report — both implemented as LangChain `@tool`s.

## Tech stack

| Layer | Tools |
|---|---|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS |
| LLMs | Groq (`llama-3.1-8b-instant`) for retrieval summaries, OpenAI (`gpt-4o-mini`) for the structured agent |
| Orchestration | LangChain (tools, agents, prompt templates) |
| Structured output | Pydantic |
| Report generation | ReportLab (PDF) |
| UI | Streamlit |

## Project structure

```
├── app.py              # Streamlit UI (entry point)
├── rag_engine.py        # Chunking, embeddings, FAISS store, RAG summarization
├── agent.py              # Structured insights schema, tools, agent executor
├── data/                 # Sample anonymized tutoring transcripts (JSON)
├── requirements.txt
├── .env.example
└── rag_faiss_pipeline.ipynb   # Original exploratory notebook
```

## Use cases

- **EdTech platforms** — personalized learning analytics at scale
- **Tutors & teachers** — automatic progress tracking across sessions
- **Learning analytics research** — engagement & comprehension trend analysis
- **Student support teams** — early identification of confusion or disengagement

## Notes

- Sample data in `data/` is small, anonymized, and included only to demo the pipeline —
  swap in your own transcripts (same `role`/`text`/`timestamp` JSON shape) to try it on
  real sessions.
- The FAISS index is rebuilt automatically the first time you run the app if `faiss_store/`
  doesn't already exist.
