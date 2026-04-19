import uuid

import streamlit as st

from legal_assistant.graph import build_app
from legal_assistant.kb import LEGAL_DOCUMENTS
from legal_assistant.state import initial_state


st.set_page_config(page_title="Legal Document Assistant", page_icon="LA", layout="wide")


@st.cache_resource
def load_agent():
    app, _, _ = build_app()
    return app


app = load_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Legal Assistant")
    st.write("Grounded Q&A for legal document review.")
    show_trace = st.checkbox("Show debug trace", value=False)
    st.write("Topics covered:")
    for doc in LEGAL_DOCUMENTS:
        st.caption(doc["topic"])
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.title("Legal Document Assistant")
st.caption("Answers are grounded in the project knowledge base and are not legal advice.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about contracts, NDAs, leases, timelines, discovery, property rights, or compliance...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = app.invoke(
        initial_state(question),
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    answer = result.get("answer", "")
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        if show_trace:
            with st.expander("Trace"):
                st.write("Route:", result.get("route"))
                st.write("Sources:", result.get("sources", []))
                st.write("Faithfulness:", result.get("faithfulness"))
                st.write("Node trace:", " -> ".join(result.get("trace", [])))
