"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi câu hoặc ý trả lời phải có citation dạng [S1], [S2], ... đúng với nhãn
Citation được gán cho nguồn trong context. Không tự tạo nhãn hoặc rút gọn ID.
Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
_CITATION_PATTERN = re.compile(r"\[([^\[\]\s]+)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Source S{index} | ID: {chunk['id']} | "
            f"Title: {metadata['title']} | Source: {metadata['source']} | "
            f"URL: {metadata.get('url') or 'not provided'}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _expand_citation_aliases(answer: str, chunks: list[dict]) -> str:
    """Expand short prompt labels back to stable chunk IDs for the API/UI."""
    labels = {str(index): chunk["id"] for index, chunk in enumerate(chunks, 1)}

    def replace(match: re.Match) -> str:
        chunk_id = labels.get(match.group(1))
        return f"[{chunk_id}]" if chunk_id is not None else match.group(0)

    return re.sub(r"\[S(\d+)\]", replace, answer, flags=re.IGNORECASE)


def citations_map_to_sources(answer: str, sources: list[dict]) -> bool:
    """Require every bracket citation in the answer to name a retrieved ID."""
    citations = _CITATION_PATTERN.findall(answer)
    if not citations:
        return False
    source_ids = {source.get("id") for source in sources}
    return all(citation in source_ids for citation in citations)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI-compatible API, Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    if provider in {"openai", "groq"}:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = os.getenv(
                "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
            ).strip() or "https://api.groq.com/openai/v1"
            default_model = "openai/gpt-oss-20b"
            key_name = "GROQ_API_KEY"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
            default_model = "gpt-4o-mini"
            key_name = "OPENAI_API_KEY"

        if not api_key:
            raise RuntimeError(f"{key_name} is required for the {provider} provider")

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model or default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    if provider in {"gemini", "google", "google-genai"}:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini provider")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model or "gemini-2.0-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return (response.text or "").strip()

    if provider in {"anthropic", "claude"}:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for the Anthropic provider"
            )

        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-3-5-haiku-latest",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        text_parts = [
            block.text
            for block in response.content
            if isinstance(getattr(block, "text", None), str)
        ]
        return "\n".join(text_parts).strip()

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}; use openai, groq, gemini or anthropic"
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    try:
        chunks = retrieve(query, top_k=top_k) or []
    except Exception:
        # Retrieval providers are optional at this layer; the UI should still
        # receive a valid GenerationResult when one is unavailable.
        chunks = []

    if not chunks:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query.strip()}"

    try:
        answer = _expand_citation_aliases(
            call_llm(SYSTEM_PROMPT, user_message).strip(), reordered
        )
    except Exception:
        answer = ""

    if not answer:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    if not citations_map_to_sources(answer, chunks):
        methods = {item.get("retrieval_method") for item in chunks}
        retrieval_source = "pageindex" if "pageindex" in methods else "hybrid"
        return {
            "answer": SAFE_REFUSAL,
            "sources": list(chunks),
            "retrieval_source": retrieval_source,
        }

    methods = {item.get("retrieval_method") for item in chunks}
    retrieval_source = "pageindex" if "pageindex" in methods else "hybrid"
    return {
        "answer": answer,
        # Keep the original score-descending order for the public contract.
        # The context carries stable chunk IDs, so reordering does not break
        # citation mapping.
        "sources": list(chunks),
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
