"""
app.py
Streamlit UI for the Teacher Intelligence Agent.

Lets an educator pick a tutoring transcript, ask a question about the student's
understanding, and get back structured insights (+ an optional downloadable PDF).

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (see README for step-by-step instructions)
"""
import os
import json
import streamlit as st

st.set_page_config(page_title="Teacher Intelligence Agent", page_icon="🧑‍🏫", layout="centered")

st.title("🧑‍🏫 Teacher Intelligence Agent")
st.caption("RAG + an LLM tool-using agent that turns tutoring transcripts into actionable student insights.")

with st.sidebar:
    st.header("Setup")
    groq_key = st.text_input("GROQ_API_KEY", type="password", value=os.getenv("GROQ_API_KEY", ""))
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    st.caption("Keys are only used for this session and are never stored.")
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Transcripts are chunked & embedded (SentenceTransformers)\n"
        "2. Stored in a FAISS vector index\n"
        "3. A LangChain agent retrieves relevant chunks (`rag_search` tool) "
        "and reasons over them\n"
        "4. Insights are returned in a structured schema, with an optional "
        "PDF export tool"
    )

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

query = st.text_area(
    "Ask about a student's session",
    placeholder="e.g. Did the student understand the three states of matter? What should the tutor do next?",
    height=90,
)

col1, col2 = st.columns(2)
run_search = col1.button("🔍 Quick RAG summary", use_container_width=True)
run_agent = col2.button("🧠 Full structured insights + PDF", use_container_width=True)

if (run_search or run_agent) and not (groq_key and (run_search or (run_agent and openai_key))):
    st.warning("Add your GROQ_API_KEY (and OPENAI_API_KEY for full insights) in the sidebar first.")
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
