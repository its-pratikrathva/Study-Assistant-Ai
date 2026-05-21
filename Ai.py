"""
Study Assistant AI — Ai.py
Fixed: dotenv loading, Gemini API key detection, improved code structure
"""

# ====================== IMPORTS ======================
import os
import sys
from pathlib import Path
from datetime import datetime

# ── CRITICAL: Load .env BEFORE any other import that might touch env vars ──
from dotenv import load_dotenv

# Get the directory where Ai.py lives and load .env from there
_BASE_DIR = Path(__file__).resolve().parent
_ENV_PATH = _BASE_DIR / ".env"

load_dotenv(dotenv_path=_ENV_PATH, override=True)

# Double-check: also expose the key under the name Google's library auto-detects
_gemini_key = os.getenv("GEMINI_API_KEY")
if _gemini_key:
    os.environ["GOOGLE_API_KEY"] = _gemini_key   # ← this is the fix
    os.environ["GEMINI_API_KEY"] = _gemini_key    # keep both in sync

import streamlit as st

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🎓 Study Assistant AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== CUSTOM CSS ======================
st.markdown(
    """
    <style>
    /* ---------- global ---------- */
    .main { background-color: #0e1117; color: #e8eaf6; }

    /* ---------- buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #00c853, #00897b);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ---------- inputs ---------- */
    .stTextArea textarea,
    .stTextInput input       { border-radius: 10px; }

    .stSelectbox div[data-baseweb="select"] { border-radius: 10px; }

    /* ---------- expanders ---------- */
    div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid #1e2a3a; }

    /* ---------- footer ---------- */
    .footer { text-align: center; color: #607d8b; padding-top: 1rem; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ====================== HELPERS ======================

def _load_gemini_key() -> str | None:
    """
    Return the Gemini API key, checking every possible source so that
    Streamlit's subprocess isolation can't hide it.
    """
    # 1. Already in environment (set at module load above)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    # 2. Re-read .env from disk as a last resort
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    os.environ["GEMINI_API_KEY"] = key
                    os.environ["GOOGLE_API_KEY"] = key
                    return key

    return None

def get_gemini_response(prompt: str, temperature: float = 0.7) -> str:
    """
    Gemini call with quota-aware model rotation.
    Rotates: gemini-2.0-flash → gemini-2.0-flash-lite → gemini-2.5-flash
    Handles: 429 quota, 503 overload, retry-delay parsing
    """
    import time
    import re
    from google import genai
    from google.genai import types

    api_key = _load_gemini_key()
    if not api_key:
        st.error("❌ API key not found in .env file.")
        return ""

    client = genai.Client(api_key=api_key)

    # ✅ Ordered by free-tier generosity (most quota → least)
    models_to_try = [
        "gemini-2.0-flash",        # 200 req/day free
        "gemini-2.0-flash-lite",   # 1500 req/day free
        "gemini-2.5-flash",        # 20 req/day free (last resort)
    ]

    last_error = ""

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                ),
            )
            # Show which model was actually used
            st.caption(f"✅ Response from: `{model_name}`")
            return response.text

        except Exception as exc:
            error_str = str(exc)
            last_error = error_str

            is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            is_503 = "503" in error_str or "UNAVAILABLE" in error_str

            if is_429:
                # Parse retry delay from error message if present
                delay_match = re.search(r'retry[^\d]*(\d+)', error_str, re.IGNORECASE)
                delay = int(delay_match.group(1)) if delay_match else 0

                st.toast(
                    f"⚠️ `{model_name}` quota exhausted "
                    f"{'— retrying in ' + str(delay) + 's' if delay else ''}"
                    f" → switching model..."
                )

                if delay and delay < 10:
                    # Only auto-wait if delay is short (don't block UI for 40s)
                    time.sleep(delay)

                continue  # try next model

            elif is_503:
                st.toast(f"⏳ `{model_name}` overloaded → switching model...")
                time.sleep(2)
                continue  # try next model

            else:
                # Unknown error — surface immediately, don't rotate
                st.error(f"Gemini Error: {exc}")
                return ""

    # All models exhausted
    st.error(
        "❌ **All Gemini models quota exhausted for today.**\n\n"
        "**Options:**\n"
        "- Wait until midnight (Pacific Time) for quota reset\n"
        "- Add billing at [aistudio.google.com](https://aistudio.google.com) for higher limits\n"
        f"\n_Last error: {last_error[:200]}_"
    )
    return ""


@st.cache_resource(show_spinner=False)
def _load_flan_pipeline(model_name: str):
    from transformers import pipeline
    return pipeline("text2text-generation", model=model_name)


def get_local_response(prompt: str, model_choice: str, max_length: int) -> str:
    model_name = (
        "google/flan-t5-large" if "Large" in model_choice else "google/flan-t5-base"
    )
    pipe = _load_flan_pipeline(model_name)
    result = pipe(prompt, max_length=max_length)
    return result[0]["generated_text"]


def generate_response(
    prompt: str,
    model_choice: str,
    temperature: float,
    max_length: int,
) -> str:
    if "Gemini" in model_choice:
        return get_gemini_response(prompt, temperature)
    return get_local_response(prompt, model_choice, max_length)


def save_to_history(task: str, model: str, user_input: str, response: str) -> None:
    st.session_state.history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "task": task,
            "model": model,
            "input": user_input[:80],
            "response": response,
        }
    )


# ====================== SESSION STATE ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ====================== HEADER ======================
st.title("🎓 Study Assistant AI")
st.markdown("### Your all-in-one AI study companion")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")

    # Show key status at a glance
    _key_present = bool(_load_gemini_key())
    if _key_present:
        st.success("✅ Gemini API key loaded")
    else:
        st.error("❌ Gemini API key missing")

    model_choice = st.selectbox(
        "Choose AI Model",
        [
            "Gemini 1.5 Flash (Recommended)",
            "FLAN-T5 Base",
            "FLAN-T5 Large",
        ],
    )

    max_length = st.slider("Max Response Length (local models)", 100, 500, 250)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05)

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared!")

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Text Tools", "📄 PDF Summarizer", "❓ MCQ Generator", "📜 History"]
)

# =========================================================
# TAB 1 — TEXT TOOLS
# =========================================================
with tab1:

    option = st.selectbox(
        "Choose Task",
        ["Explain Topic", "Summarize Text", "Short Answer", "Essay Style"],
    )

    user_input = st.text_area("Enter your text or topic:", height=200)

    if st.button("🚀 Generate Response", key="btn_text"):

        if not user_input.strip():
            st.warning("⚠️ Please enter some text first.")

        else:
            prompts = {
                "Explain Topic": f"""
Explain this topic in very simple, student-friendly language.

Include:
- Clear, easy explanation
- Bullet points for key ideas
- Real-world examples
- Important concepts to remember

Topic:
{user_input}
""",
                "Summarize Text": f"""
Summarize the following text clearly and concisely using bullet points.

Text:
{user_input}
""",
                "Essay Style": f"""
Write a detailed, well-structured essay on the following topic.
Include an introduction, body paragraphs, and a conclusion.

Topic:
{user_input}
""",
                "Short Answer": f"""
Answer the following question briefly and accurately in 2-4 sentences:

{user_input}
""",
            }

            with st.spinner("🤖 AI is thinking..."):
                reply = generate_response(
                    prompts[option], model_choice, temperature, max_length
                )

            if reply:
                save_to_history(option, model_choice, user_input, reply)
                st.subheader("✅ AI Response")
                st.write(reply)

# =========================================================
# TAB 2 — PDF SUMMARIZER
# =========================================================
with tab2:

    st.subheader("📄 Upload PDF Notes")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        try:
            from pypdf import PdfReader

            with st.spinner("📖 Reading PDF..."):
                reader = PdfReader(uploaded_file)
                pdf_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )

            char_count = len(pdf_text)
            st.success(f"✅ Extracted {char_count:,} characters from {len(reader.pages)} page(s)")

            if char_count == 0:
                st.warning("⚠️ No text could be extracted. The PDF may be scanned/image-based.")
            else:
                if st.button("📝 Summarize PDF", key="btn_pdf"):
                    with st.spinner("🤖 Summarizing..."):
                        prompt = f"""
Summarize these study notes for exam preparation.

Include:
- Main concepts
- Important points
- Bullet-point summary
- Key terms and definitions

Notes:
{pdf_text[:5000]}
"""
                        summary = get_gemini_response(prompt, temperature=0.5)

                    if summary:
                        st.subheader("📋 Summary")
                        st.write(summary)
                        st.download_button(
                            "💾 Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain",
                        )

        except Exception as exc:
            st.error(f"Error reading PDF: {exc}")

# =========================================================
# TAB 3 — MCQ GENERATOR
# =========================================================
with tab3:

    st.subheader("❓ MCQ Generator")
    topic = st.text_input("Enter Topic")
    num_questions = st.slider("Number of Questions", 3, 10, 5)

    if st.button("🎯 Generate MCQs", key="btn_mcq"):

        if not topic.strip():
            st.warning("⚠️ Please enter a topic.")

        else:
            prompt = f"""
Generate {num_questions} high-quality multiple choice questions on: {topic}

Use this exact format for every question:

Q1. <question text>
A) <option>
B) <option>
C) <option>
D) <option>

✅ Correct Answer: <letter>
💡 Explanation: <brief explanation>

---
"""
            with st.spinner("🤖 Creating MCQs..."):
                mcqs = get_gemini_response(prompt, temperature=0.8)

            if mcqs:
                save_to_history("MCQ Generator", model_choice, topic, mcqs[:150])
                st.write(mcqs)

# =========================================================
# TAB 4 — HISTORY
# =========================================================
with tab4:

    st.subheader("📜 Activity History")

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(
                f"[{item['timestamp']}] {item['task']} — {item['model']}"
            ):
                st.markdown("**Input:**")
                st.write(item["input"])
                st.markdown("**Response:**")
                st.write(item["response"])
    else:
        st.info("No activity yet. Start by using the Text Tools or MCQ Generator!")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    '<p class="footer">Made with ❤️ using Streamlit + Gemini AI</p>',
    unsafe_allow_html=True,
)