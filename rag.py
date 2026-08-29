"""
rag.py
------
This is the heart of the RAG pipeline. It:
  1. Takes the user's report text (the QUERY)
  2. Uses the Retriever to search the vector database (RETRIEVAL)
  3. Builds a prompt containing the report + retrieved context (AUGMENTATION)
  4. Sends that prompt to the LLM (GENERATION)
  5. Returns the explanation, plus the retrieved chunks for transparency

CONTEXT = the retrieved medical knowledge text that gets inserted into the
prompt so the LLM can "read" it before answering. This is what keeps the
LLM GROUNDED (answering based on real reference material) instead of
HALLUCINATING (making up plausible-sounding but unverified information).
"""

import os
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
from vector_store import Retriever

load_dotenv()

api_key = st.secrets["ANTHROPIC_API_KEY"]
client = Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are an educational medical report assistant.

Explain the provided medical report using ONLY the relevant information
retrieved from the medical knowledge base below. Do not use outside
medical knowledge beyond what is given in the retrieved context.

Instructions:
1. Explain the report in simple, plain language a non-expert can understand.
2. Identify which values appear potentially abnormal, based on the retrieved
   reference ranges (if given).
3. Explain what those values can sometimes be associated with, using the
   retrieved context.
4. Do NOT provide a definitive diagnosis.
5. Do NOT invent medical information that is not in the retrieved context.
6. If the retrieved context is insufficient to explain a value, say so plainly.
7. Recommend consulting a qualified healthcare professional when appropriate.
8. Remind the user that laboratory reference ranges can vary between labs
   and individuals.

This is an educational explanation, not a medical diagnosis."""

USER_PROMPT_TEMPLATE = """Medical Report:
{report}

Retrieved Medical Information:
{context}

Please provide:
1. Medical Report Summary
2. Important Findings
3. AI Explanation
4. Recommendation"""


def format_context(retrieved_chunks):
    """Turn the list of retrieved chunks into one text block for the prompt."""
    parts = []
    for chunk in retrieved_chunks:
        parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def run_rag_pipeline(report_text: str, top_k: int = 3) -> dict:
    """
    Runs the full RAG pipeline and returns a dictionary with everything,
    including the intermediate steps, so the app can display them for
    demonstration purposes.
    """
    # --- RETRIEVAL ---
    retriever = Retriever()
    retrieved_chunks = retriever.search(report_text, top_k=top_k)
    context_text = format_context(retrieved_chunks)

    # --- AUGMENTATION ---
    user_prompt = USER_PROMPT_TEMPLATE.format(report=report_text, context=context_text)

    # --- GENERATION ---
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    explanation = response.content[0].text

    return {
        "query": report_text,
        "retrieved_chunks": retrieved_chunks,
        "context": context_text,
        "prompt": user_prompt,
        "answer": explanation
    }
