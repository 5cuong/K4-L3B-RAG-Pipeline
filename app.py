import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Hỏi đáp về bảo vệ người tiêu dùng và thương mại điện tử")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("RAG Chatbot")
st.caption("Câu trả lời dựa trên luật và bài viết đã được lập chỉ mục")


def render_sources(result: dict) -> None:
    """Hiển thị nguồn và điểm retrieval của một câu trả lời."""
    st.caption(f"Retrieval: `{result['retrieval_source']}`")
    sources = result.get("sources", [])
    if not sources:
        return

    st.subheader("Nguồn")
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("title") or metadata.get("source") or "Tài liệu"
        origin = metadata.get("source", "")
        score = source.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
        with st.expander(f"[{index}] {title} — score {score_text}"):
            st.write(f"Source: {origin}")
            st.write(f"Chunk ID: {source.get('id', '')}")
            st.write(source.get("content", ""))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            render_sources(message["result"])

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = generate_with_citation(query, top_k)
        st.markdown(result["answer"])
        render_sources(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )
