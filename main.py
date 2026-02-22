# app.py
import os
import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from typing import Literal
from pydantic import BaseModel, Field

# -----------------------------
# Your base code (kept intact)
# -----------------------------
load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model1 = ChatHuggingFace(llm=llm1)


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
classifier_chain = prompt1 | model1 | parser2

pos_prompt = PromptTemplate(
    template="Generate a response to the positive feedback -> \n {feedback}",
    input_variables=["feedback"],
)

neg_prompt = PromptTemplate(
    template="Generate a response to the negative feedback -> \n {feedback}",
    input_variables=["feedback"],
)

# We keep BOTH the original text and the label by using assign()
chain = (
    RunnablePassthrough.assign(label=classifier_chain)
    | RunnableBranch(
        (lambda x: x["label"].sentiment == "positive", pos_prompt | model1 | StrOutputParser()),
        (lambda x: x["label"].sentiment == "negative", neg_prompt | model1 | StrOutputParser()),
        RunnableLambda(lambda x: "Thank you for your feedback!"),
    )
)

# -----------------------------
# Streamlit UI wrapper
# -----------------------------
st.set_page_config(page_title="Sentiment Feedback Assistant", page_icon="💬", layout="centered")
st.title("💬 Sentiment Feedback Assistant")
st.caption("Enter feedback text → classify sentiment → generate a response")

# Sidebar info / env check
with st.sidebar:
    st.subheader("Settings")
    hf_ok = bool(os.getenv("HF_TOKEN"))
    st.write("HF_TOKEN loaded:", "✅" if hf_ok else "❌")
    st.divider()
    st.write("Model:", "meta-llama/Llama-3.1-8B-Instruct")

# Input
text = st.text_area(
    "Feedback text",
    value="very very good product.",
    height=120,
    placeholder="Type feedback here...",
)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Analyze & Respond", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_label", None)
    st.rerun()

if run_btn:
    if not text.strip():
        st.warning("Please enter some feedback text.")
    else:
        with st.spinner("Running chain..."):
            try:
                # same invocation style as your base code
                result = chain.invoke({"text": text, "feedback": text})
                # Also run classifier alone to show the sentiment label (no change to chain)
                label = classifier_chain.invoke({"text": text})

                st.session_state["last_result"] = result
                st.session_state["last_label"] = label.sentiment
            except Exception as e:
                st.error("Error while running the model/chain.")
                st.exception(e)

# Display
if "last_result" in st.session_state:
    st.subheader("Result")
    st.markdown(f"**Predicted sentiment:** `{st.session_state.get('last_label', 'unknown')}`")
    st.text_area("Generated response", value=st.session_state["last_result"], height=140)

    st.divider()
    st.subheader("Rate this response")
    # Optional rating UI (doesn't affect your chain)
    rating = st.feedback("thumbs")
    if rating is not None:
        st.success("Thanks! Feedback recorded in this session.")
