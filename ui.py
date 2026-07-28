import streamlit as st
import requests, os

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("RAG Research Assistant")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

def fetch_documents():
    try:
        res = requests.get(f"{API}/documents")
        if res.status_code == 200:
            st.session_state.documents = res.json().get("documents", [])
    except Exception as e:
        st.sidebar.error(f"Could not fetch documents: {e}")

# Sidebar
with st.sidebar:
    st.header("Upload Paper")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded:
        if st.button("Ingest"):
            with st.spinner("Ingesting..."):
                res = requests.post(
                    f"{API}/upload_file",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                )
                if res.status_code == 200:
                    st.success(f"Ingested {res.json()['chunks_ingested']} chunks")
                    fetch_documents()
                elif res.status_code == 409:
                    st.warning("Already ingested.")
                else:
                    st.error(res.json().get("detail", "Error"))

    st.divider()
    st.header("Ingested Papers")
    fetch_documents()

    titles = [d.get("title", d["source"]) for d in st.session_state.documents]
    filter_doc = st.selectbox("Filter by paper", ["All Papers"] + titles)

    for doc in st.session_state.documents:
        title = doc.get("title", doc["source"])
        col1, col2 = st.columns([4, 1])
        col1.write(title)
        if col2.button("🗑", key=doc["source"]):
            name = doc["source"].split("\\")[-1]
            res = requests.delete(f"{API}/documents/{name}")
            if res.status_code == 200:
                st.success(f"Deleted {name}")
                fetch_documents()
                st.rerun()
            else:
                st.error(res.json().get("detail", "Error"))

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("Citations"):
                for i, c in enumerate(msg["citations"], 1):
                    st.markdown(f"**[{i}] {c['title']}**  \nPages: {c['first_page']} – {c['last_page']}  \n`{c['source']}`")

# Chat input
if question := st.chat_input("Ask a question about your papers..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {"question": question, "conversation_history": []}
    if filter_doc != "All Papers":
        matched = next((d for d in st.session_state.documents if d.get("title") == filter_doc), None)
        if matched:
            payload["filter_document"] = matched["source"].split("\\")[-1]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API}/query", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])
                    st.markdown(answer)
                    if citations:
                        with st.expander("Citations"):
                            for i, c in enumerate(citations, 1):
                                st.markdown(f"**[{i}] {c['title']}**  \nPages: {c['first_page']} – {c['last_page']}  \n`{c['source']}`")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error(res.json().get("detail", "Error"))
            except Exception as e:
                st.error(f"Could not reach API: {e}")