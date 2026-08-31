"""
app.py
Streamlit UI for the Teacher Intelligence Agent.

Lets an educator pick a tutoring transcript, ask a question about the student's
understanding, and get back structured insights (+ an optional downloadable PDF).

This is a public live demo: the GROQ_API_KEY is configured once by the app owner
(via Streamlit secrets / env var), so visitors don't need a key of their own.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (see README for step-by-step instructions)
"""
import os
import json
import streamlit as st

st.set_page_config(page_title="Teacher Intelligence Agent", page_icon="🧑‍🏫", layout="centered")


def _get_groq_key() -> str:
    """Resolves the shared demo Groq key from the environment or Streamlit secrets,
    so visitors never have to supply their own."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


groq_key = _get_groq_key()
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

st.title("🧑‍🏫 Teacher Intelligence Agent")
st.caption("RAG + an LLM tool-using agent that turns tutoring transcripts into actionable student insights.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Transcripts are chunked & embedded (SentenceTransformers)\n"
        "2. Stored in a FAISS vector index\n"
        "3. A LangChain agent retrieves relevant chunks (`rag_search` tool) "
        "and reasons over them\n"
        "4. Insights are returned in a structured schema, with an optional "
        "PDF export tool"
    )
    st.divider()
    st.caption("This is a shared live demo — no API key needed to try it.")

query = st.text_area(
    "Ask about a student's session",
    placeholder="e.g. Did the student understand the three states of matter? What should the tutor do next?",
    height=90,
)

col1, col2 = st.columns(2)
run_search = col1.button("🔍 Quick RAG summary", use_container_width=True)
run_agent = col2.button("🧠 Full structured insights + PDF", use_container_width=True)

if (run_search or run_agent) and not groq_key:
    st.error(
        "This demo isn't configured yet: the app owner needs to set GROQ_API_KEY "
        "in Streamlit secrets (or the environment)."
    )
elif run_search and query:
    with st.spinner("Retrieving relevant transcript chunks and summarizing..."):
        from rag_engine import RAGSearch
        rag = RAGSearch()
        st.markdown("### Summary")
        st.write(rag.search_and_summarize(query))

elif run_agent and query:
    with st.spinner("Running the tutoring-insights agent (retrieval → reasoning → structured output)..."):
        from agent import build_agent_executor
        executor = build_agent_executor()
        full_query = query + " Also save the insights as a PDF named 'tutoring_insights.pdf'."
        result = executor.invoke({"input": full_query})
        st.markdown("### Agent Output")
        st.write(result["output"])

        pdf_path = "tutoring_insights.pdf"
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download PDF report", f, file_name="tutoring_insights.pdf")

st.divider()
st.caption(
    "Demo data: anonymized sample math/science tutoring transcripts included in `data/`. "
    "Swap in your own JSON transcripts (role/text/timestamp turns) to try it on real sessions."
)
