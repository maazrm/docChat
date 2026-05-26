import logging
import hashlib
import streamlit as st
from dotenv import load_dotenv

from pipeline.agents.preprocessor import ingest
from pipeline.graph import build_graph
from pipeline.state import PipelineState

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Page config
st.set_page_config(
    page_title="VMARP — Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 VMARP — Document Q&A")
st.caption("Upload a PDF, ask questions, get grounded and validated answers.")

# Session state defaults
if "doc_id"        not in st.session_state: st.session_state.doc_id        = None
if "topic_summary" not in st.session_state: st.session_state.topic_summary = None
if "file_hash"     not in st.session_state: st.session_state.file_hash     = None
if "graph"         not in st.session_state: st.session_state.graph         = build_graph()
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []

# Document upload
uploaded_file = st.file_uploader("Upload a PDF (digital or scanned)", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash  = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.file_hash:
        # New file uploaded — run ingestion
        with st.spinner(
            "Processing document... "
            "(First run downloads Docling layout models ~1–2 GB. "
            "Subsequent runs are fast.)"
        ):
            result = ingest(file_bytes, uploaded_file.name)

        st.session_state.doc_id        = result["doc_id"]
        st.session_state.topic_summary = result["topic_summary"]
        st.session_state.file_hash     = file_hash
        st.session_state.chat_history  = []  # reset history for new doc

        if result["chunk_count"] > 0:
            st.success(f"✅ Document indexed — {result['chunk_count']} chunks created.")
        else:
            st.info("ℹ️ Document was already indexed. Loaded from cache.")

    # Show document info
    st.markdown(f"**Document loaded:** {uploaded_file.name}")
    st.markdown(f"**Topic:** {st.session_state.topic_summary}")
    st.divider()

# Query input
if st.session_state.doc_id:
    query = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. What were the key findings in section 3?"
    )

    if st.button("Ask") and query.strip():
        initial_state: PipelineState = {
            "doc_id":            st.session_state.doc_id,
            "topic_summary":     st.session_state.topic_summary,
            "raw_query":         query,
            "rewritten_queries": [],
            "scope_status":      "",
            "scope_reason":      "",
            "retrieved_chunks":  [],
            "raw_answer":        "",
            "claim_verdicts":    [],
            "validation_passed": False,
            "retry_count":       0,
            "final_answer":      "",
            "sources":           [],
            "status":            ""
        }

        with st.spinner("Thinking..."):
            final_state = st.session_state.graph.invoke(initial_state)

        # Display answer
        status = final_state.get("status", "")

        if status == "out_of_scope":
            st.warning(f"🚫 {final_state['final_answer']}")

        elif status == "fallback":
            st.warning(f"⚠️ {final_state['final_answer']}")

        else:
            st.markdown("### Answer")
            st.markdown(final_state["final_answer"])

            # Sources expander
            with st.expander("📎 Sources used"):
                for src in final_state.get("sources", []):
                    st.markdown(
                        f"- **Page {src['page']}** · `{src['chunk_type']}` — "
                        f"*{src['text_snippet']}*"
                    )

            # Validation details expander
            verdicts = final_state.get("claim_verdicts", [])
            retry_count = final_state.get("retry_count", 0)

            with st.expander("🔍 Validation details"):
                if retry_count == 0:
                    st.markdown("**Status:** ✅ Passed on first attempt")
                else:
                    st.markdown(f"**Status:** ✅ Passed after {retry_count} retry attempt(s)")

                supported   = sum(1 for v in verdicts if v["verdict"] == "supported")
                unsupported = sum(1 for v in verdicts if v["verdict"] == "unsupported")
                st.markdown(f"**Claims checked:** {supported} supported · {unsupported} unsupported")

                for v in verdicts:
                    icon = "✅" if v["verdict"] == "supported" else "❌"
                    st.markdown(f"{icon} **{v['claim']}**  \n*{v['reason']}*")

        # Save to chat history
        st.session_state.chat_history.append({
            "query":  query,
            "answer": final_state.get("final_answer", ""),
            "status": status
        })

    # Chat history
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### Chat history")
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            status_icon = {"success": "✅", "fallback": "⚠️", "out_of_scope": "🚫"}.get(item["status"], "✅")
            with st.expander(f"{status_icon} Q: {item['query']}"):
                st.markdown(item["answer"])

else:
    st.info("⬆️ Upload a PDF above to get started.")
