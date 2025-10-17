"""
===============================================================================
💻 AI Code Assistant
-------------------------------------------------------------------------------
Author: [Jaimin Desai]
License: CC BY-NC-ND 4.0 (Creative Commons Attribution–NonCommercial–NoDerivatives)
Repository: [Your GitHub Repo URL]

🧠 Project Note:
This project demonstrates my exploration of AI-assisted development — leveraging
large language models for rapid prototyping while maintaining code ownership and
responsible disclosure.

The AI model used here is locally served via an LLM interface but its exact
identity and configuration are intentionally undisclosed for protection.
Ownership and rights to this script rest entirely with the author.

⚠️ Liability Disclaimer:
This software is provided “as is,” without warranty of any kind.
The author assumes no responsibility for errors, bugs, or any damages
arising from the use of this script or its outputs.
===============================================================================
"""

import streamlit as st
import time
import requests
from langchain_ollama import OllamaLLM

# --- Model config ---
MODEL_NAME = "local-llm"  # Model identity intentionally abstracted for publication

# --- Cache model loading ---
@st.cache_resource
def load_llm():
    """Load and cache the local LLM model with specified parameters."""
    st.markdown("### 🤖 Initializing AI Code Assistant...")
    return OllamaLLM(
        model=MODEL_NAME,
        temperature=0.3,
        top_p=0.95,
        max_tokens=8192,
        stop=None
    )

# --- Streamlit UI setup ---
st.set_page_config(page_title="AI Code Assistant", page_icon="💻", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stTextArea textarea { font-family: 'Courier New', Courier, monospace; }
    </style>
""", unsafe_allow_html=True)
st.title("💻 AI Code Assistant")

# --- Load model ---
llm = load_llm()

# --- Prompt generation map ---
prompt_map = {
    "🧠 Explain code": lambda q: f"Please explain the following code:\n```csharp\n{q}\n```",
    "🔍 Review code for bugs": lambda q: f"Please review the following code and identify any bugs or issues:\n```csharp\n{q}\n```",
    "✨ Improve or refactor code": lambda q: f"Refactor or improve this code for better readability and performance:\n```csharp\n{q}\n```",
    "💬 Ask general coding question": lambda q: q
}

# --- Initialize session state for text input ---
if 'query' not in st.session_state:
    st.session_state.query = ""

# --- Text input and character count ---
MAX_CHARS = 5000
st.session_state.query = st.text_area(
    "Paste code or ask your question below:",
    value=st.session_state.query,
    height=250,
    max_chars=MAX_CHARS,
    key="query_input"
)
st.markdown(f"**{len(st.session_state.query)}/{MAX_CHARS} characters used**")

# --- Form for submission ---
with st.form("code_form"):
    mode = st.radio("What do you want to do?", [
        "🧠 Explain code",
        "🔍 Review code for bugs",
        "✨ Improve or refactor code",
        "💬 Ask general coding question"
    ])
    submitted = st.form_submit_button("🚀 Submit")

# --- Handle submission ---
if submitted:
    if not st.session_state.query.strip():
        st.warning("⚠️ Please provide a valid code snippet or question.")
    elif len(st.session_state.query) > MAX_CHARS:
        st.warning(f"⚠️ Input is too long. Please limit to {MAX_CHARS} characters.")
    else:
        prompt = prompt_map[mode](st.session_state.query)
        st.markdown(f"🧮 Prompt length: `{len(prompt.split())}` words (approx token estimate)")

        with st.spinner("💡 Thinking..."):
            try:
                start = time.time()
                response = llm(prompt)
                elapsed = time.time() - start

                st.markdown("### ✅ Response:")
                if mode in ["💬 Ask general coding question", "🧠 Explain code"]:
                    st.markdown(response)
                else:
                    st.code(response, language="csharp")

                st.markdown(f"⏱️ _Answered in `{elapsed:.2f}` seconds._")
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to connect to the model server. Please check if the LLM is running locally.")
            except ValueError as ve:
                st.error(f"❌ Invalid input or model configuration: {str(ve)}")
            except Exception as e:
                st.error("❌ An unexpected error occurred during model inference.")
                st.exception(e)
