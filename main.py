# app.py
import os
import streamlit as st

from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough


# -----------------------------
# Page config (Streamlit)
# -----------------------------
st.set_page_config(page_title="Sentiment Feedback Assistant", page_icon="💬", layout="centered")
st.title("💬 Sentiment Feedback Assistant")
st.caption("Select a model → enter feedback → classify sentiment → generate a response")


# -----------------------------
# Secrets / env (secure token)
# -----------------------------
load_dotenv()  # local dev support

# Priority:
# 1) Streamlit secrets (recommended for deployment)
# 2) Environment variable (local/.env)
HF_TOKEN = None
if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN  # keep your base pattern
else:
    st.error("HF_TOKEN not found. Add it to Streamlit Secrets or local .env.")
    st.stop()


# -----------------------------
# Model selection
# -----------------------------
st.sidebar.subheader("Model Selection")

ALLOWED_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

use_custom = st.sidebar.checkbox("Use custom model repo_id", value=False)

if use_custom:
    selected_repo_id = st.sidebar.text_input(
        "Custom repo_id",
        value="meta-llama/Llama-3.1-8B-Instruct",
        help="Use format: org/model. Must support text-generation.",
    ).strip()

    # Light validation (avoid obvious bad input)
    if "/" not in selected_repo_id or " " in selected_repo_id:
        st.sidebar.error("Invalid repo_id format. Example: meta-llama/Llama-3.1-8B-Instruct")
        st.stop()
else:
    selected_repo_id = st.sidebar.selectbox("Choose a model", ALLOWED_MODELS, index=0)

st.sidebar.caption(f"Selected: `{selected_repo_id}`")


# -----------------------------
# Your base code (same logic, but model is dynamic)
# -----------------------------
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the text"
    )
    # reason: str = Field(description="A brief explanation of why this sentiment was chosen")


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        "You are a strict JSON generator.\n"
        "Return ONLY a valid JSON object (no prose, no code fences, no markdown).\n"
        "The JSON MUST match this schema:\n"
        "{format_instructions}\n\n"
        "Text: {text}\n"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

pos_prompt = PromptTemplate(
    template="Generate a response to the positive feedback -> \n {feedback}",
    input_variables=["feedback"],
)

neg_prompt = PromptTemplate(
    template="Generate a response to the negative feedback -> \n {feedback}",
    input_variables=["feedback"],
)


@st.cache_resource(show_spinner=False)
def build_chain(repo_id: str):
    llm1 = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation")
    model1 = ChatHuggingFace(llm=llm1)

    classifier_chain = prompt1 | model1 | parser2

    chain = (
        RunnablePassthrough.assign(label=classifier_chain)
        | RunnableBranch(
            (lambda x: x["label"].sentiment == "positive", pos_prompt | model1 | StrOutputParser()),
            (lambda x: x["label"].sentiment == "negative", neg_prompt | model1 | StrOutputParser()),
            RunnableLambda(lambda x: "Thank you for your feedback!"),
        )
    )
    return chain, classifier_chain


# Build chain for selected model
try:
    chain, classifier_chain = build_chain(selected_repo_id)
except Exception as e:
    st.error("Failed to initialize the selected model/chain.")
    st.exception(e)
    st.stop()


# -----------------------------
# UI: Input + Run
# -----------------------------
text = st.text_area(
    "Feedback text",
    value="very very good product.",
    height=120,
    placeholder="Type feedback here...",
)

col1, col2 = st.columns([1, 1])
run_btn = col1.button("Analyze & Respond", type="primary", use_container_width=True)
clear_btn = col2.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_label", None)
    st.rerun()

if run_btn:
    if not text.strip():
        st.warning("Please enter some feedback text.")
    else:
        with st.spinner("Running..."):
            try:
                # Same invocation style as your base code
                result = chain.invoke({"text": text, "feedback": text})
                label_obj = classifier_chain.invoke({"text": text})

                st.session_state["last_result"] = result
                st.session_state["last_label"] = label_obj.sentiment
            except Exception as e:
                st.error("Error while running the chain.")
                st.exception(e)


# -----------------------------
# UI: Output
# -----------------------------
if "last_result" in st.session_state:
    st.subheader("Result")
    st.markdown(f"**Predicted sentiment:** `{st.session_state.get('last_label', 'unknown')}`")
    st.text_area("Generated response", value=st.session_state["last_result"], height=160)
