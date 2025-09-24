
import io
import os
import re
import time
from typing import List, Tuple

import streamlit as st

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

try:
    import faiss 
    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except Exception:
    ST_OK = False

try:
    import docx2txt
    DOCX_OK = True
except Exception:
    DOCX_OK = False

try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False

from openai import OpenAI


def get_api_key():
    """Ask user to enter API key directly in Streamlit if not already set"""
    if "api_key" not in st.session_state or not st.session_state.api_key:
        st.sidebar.subheader("🔑 API Key Required")
        api_key = st.sidebar.text_input("Paste your OpenAI API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
    return st.session_state.get("api_key", None)


def init_openai_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key or not api_key.startswith("sk-"):
        st.error("❌ Please enter a valid API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)

def apply_theme(dark: bool):
    css = """
    <style>
      :root {
        --bg: #ffffff; /* light bg */
        --fg: #0b0f19; /* light text */
        --card: #f4f6fb;
        --bubble-user: #e6eefb;
        --bubble-assistant: #edf0f7;
      }
      .dark-mode {
        --bg: #0b0f19; /* dark bg */
        --fg: #eaeef8; /* dark text */
        --card: #141a29;
        --bubble-user: #2a3a5a;
        --bubble-assistant: #182238;
      }

      body, .stApp { background: var(--bg) !important; color: var(--fg) !important; }
      .stMarkdown, .stText, label, p, span { color: var(--fg) !important; }

      /* Inputs, selects, textareas */
      input, textarea, select {
        background-color: var(--card) !important;
        color: var(--fg) !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
      }

      /* Chat input */
      [data-baseweb="textarea"] textarea {
        background-color: var(--card) !important;
        color: var(--fg) !important;
      }

      /* Buttons */
      button, .stButton>button {
        background-color: #4f6ef7 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
      }

      /* Expander headers */
      .streamlit-expanderHeader {
        background-color: var(--card) !important;
        color: var(--fg) !important;
      }

      /* Chat bubbles */
      [data-testid="stChatMessage"]:has(div[data-testid="user-avatar"]) {
        background: var(--bubble-user) !important; color: var(--fg) !important; border-radius: 16px; padding: 8px 12px;
      }
      [data-testid="stChatMessage"]:has(div[data-testid="assistant-avatar"]) {
        background: var(--bubble-assistant) !important; color: var(--fg) !important; border-radius: 16px; padding: 8px 12px;
      }

      .im-card { background: var(--card); padding: 12px; border-radius: 14px; }
      .im-footer { text-align:center; opacity: 0.75; margin-top: 24px; color: var(--fg) !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    if dark:
        st.markdown('<div class="dark-mode"></div>', unsafe_allow_html=True)


def split_text(text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def load_files_and_build_index(files) -> Tuple[object, object]:
    """Returns (index, embedder) or (None, None) if unavailable."""
    if not (FAISS_OK and ST_OK):
        return None, None
    texts: List[str] = []
    for f in files or []:
        if f.type == "text/plain":
            texts.append(f.read().decode("utf-8", errors="ignore"))
        elif f.type in ("application/pdf",) and PDF_OK:
            reader = PdfReader(f)
            buff = []
            for page in reader.pages:
                try:
                    buff.append(page.extract_text() or "")
                except Exception:
                    pass
            texts.append("\n".join(buff))
        elif f.type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",) and DOCX_OK:
            with open(f.name, "wb") as temp_out:
                temp_out.write(f.getvalue())
            try:
                texts.append(docx2txt.process(f.name) or "")
            finally:
                try:
                    os.remove(f.name)
                except Exception:
                    pass
        else:
            st.warning(f"Unsupported file type or missing parser: {f.name}")

    merged = "\n\n".join(t.strip() for t in texts if t and t.strip())
    if not merged:
        return None, None

    docs = split_text(merged)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.encode(docs, show_progress_bar=False)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    st.session_state["rag_docs"] = docs
    return index, embedder


def retrieve_context(query: str, index, embedder, top_k: int = 4) -> str:
    if not (index and embedder):
        return ""
    qv = embedder.encode([query])
    faiss.normalize_L2(qv)
    D, I = index.search(qv, top_k)
    docs = st.session_state.get("rag_docs", [])
    parts = []
    for idx in I[0]:
        if 0 <= idx < len(docs):
            parts.append(docs[idx])
    context = "\n\n".join(parts)
    return context


def estimate_tokens(text: str) -> int:
    words = len(re.findall(r"\w+", text))
    return max(1, int(words / 0.75))


def export_chat_txt(messages: List[dict]) -> bytes:
    lines = []
    for m in messages:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines).encode("utf-8")


def export_chat_pdf(messages: List[dict]) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed")
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x_margin, y_margin = 40, 40
    y = height - y_margin
    c.setFont("Helvetica", 11)

    def draw_wrapped(text: str, prefix: str = ""):
        nonlocal y
        max_width = width - 2 * x_margin
        words = text.split()
        line = prefix
        for w in words:
            trial = (line + " " + w).strip()
            if c.stringWidth(trial, "Helvetica", 11) > max_width:
                c.drawString(x_margin, y, line)
                y -= 16
                if y < y_margin:
                    c.showPage(); c.setFont("Helvetica", 11); y = height - y_margin
                line = w
            else:
                line = trial
        if line:
            c.drawString(x_margin, y, line)
            y -= 16
            if y < y_margin:
                c.showPage(); c.setFont("Helvetica", 11); y = height - y_margin

    c.drawString(x_margin, y, "Imtiaz Chat Export")
    y -= 24
    for m in messages:
        role = m.get("role", "assistant").capitalize()
        draw_wrapped(f"{role}:")
        draw_wrapped(m.get("content", ""))
        y -= 8
        if y < y_margin:
            c.showPage(); c.setFont("Helvetica", 11); y = height - y_margin
    c.save()
    buf.seek(0)
    return buf.read()


st.set_page_config(page_title="My ChatBot", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark mode", value=True)
    language = st.selectbox("🌍 Language", ["English", "Urdu", "Hindi", "Arabic", "French"], index=0)
    system_style = st.text_area("🧠 System Prompt / Style", value=(
        "You are a helpful, concise assistant. Keep answers clear."
    ), height=80)
    model_name = st.selectbox("🧩 Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)

apply_theme(dark_mode)

client = init_openai_client()

st.title("🤖 My Chatbot")

with st.expander("📄 Knowledge Base (Upload files for context)", expanded=False):
    rag_files = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        build_btn = st.button("🔧 Build/Refresh Index", use_container_width=True)
    with col_u2:
        clear_kb = st.button("🗑️ Clear KB", use_container_width=True)

    if clear_kb:
        st.session_state.pop("rag_index", None)
        st.session_state.pop("rag_embedder", None)
        st.session_state.pop("rag_docs", None)
        st.success("Knowledge base cleared.")

    if build_btn:
        if not (FAISS_OK and ST_OK):
            st.warning("FAISS/SentenceTransformers not available. Install 'faiss-cpu' and 'sentence-transformers'.")
        else:
            with st.spinner("Building index..."):
                idx, emb = load_files_and_build_index(rag_files)
                if idx is not None:
                    st.session_state["rag_index"] = idx
                    st.session_state["rag_embedder"] = emb
                    st.success("Index ready ✅")
                else:
                    st.warning("No text extracted from files.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))

user_prompt = st.chat_input("Type your message…")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    lang_prompts = {
        "English": "Respond in English.",
        "Urdu": "Urdu Roman me respond karo, simple aur friendly.",
        "Hindi": "Hindi (Devanagari) me respond karein, seedha aur friendly.",
        "Arabic": "Respond in Arabic.",
        "French": "Répondez en français de manière concise.",
    }

    final_system = f"{system_style}\n\n{lang_prompts.get(language, '')}"

    ctx = ""
    if st.session_state.get("rag_index") is not None:
        ctx = retrieve_context(user_prompt, st.session_state.get("rag_index"), st.session_state.get("rag_embedder"), top_k=4)
        if ctx:
            final_system += "\n\nUse the following context if relevant. If not relevant, ignore it.\n---\n" + ctx

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                msgs = [{"role": "system", "content": final_system}] + st.session_state.messages
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"(Error: {e})"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Stats")
    total_msgs = len(st.session_state.messages)
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in st.session_state.messages)
    st.write(f"Messages: **{total_msgs}**")
    st.write(f"Token estimate: **{total_tokens}**")

    st.markdown("---")
    st.subheader("🧹 Utilities")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.experimental_rerun()

    txt_bytes = export_chat_txt(st.session_state.messages)
    st.download_button("⬇️ Download Chat (TXT)", data=txt_bytes, file_name="chat_export.txt", mime="text/plain", use_container_width=True)

    if REPORTLAB_OK:
        try:
            pdf_bytes = export_chat_pdf(st.session_state.messages)
            st.download_button("⬇️ Download Chat (PDF)", data=pdf_bytes, file_name="chat_export.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.info("PDF export not available: " + str(e))
    else:
        st.info("Install 'reportlab' to enable PDF export.")

st.markdown("<div class='im-footer'>© 2025 • Built by <b>Imtiaz Hussain</b></div>", unsafe_allow_html=True)
