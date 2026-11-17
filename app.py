import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="FastAPI Docs Q&A", page_icon="📚")
st.title("📚 FastAPI Docs Q&A")
st.caption("Answers come only from the FastAPI docs, with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    for s in sources:
        with st.expander(f"[{s['n']}] {s['title']} › {s['heading']}"):
            st.markdown(f"[{s['url']}]({s['url']})")
            st.text(s["snippet"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if prompt := st.chat_input("Ask about FastAPI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            try:
                resp = requests.post(API_URL, json={"question": prompt}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()
        st.markdown(data["answer"])
        show_sources(data["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
    )