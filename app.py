"""
app.py
------
The Streamlit frontend. This ties together:

IMAGE PATH:
Image -> OCR (ocr.py) -> Extracted Text -> RAG (rag.py) -> Explanation

TEXT PATH:
Typed Text -> RAG (rag.py) -> Explanation

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from ocr import extract_text_from_image
from rag import run_rag_pipeline
from vector_store import build_index, INDEX_PATH, META_PATH

# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------
st.set_page_config(page_title="Medical Report RAG Assistant", page_icon="🩺")

# Build the FAISS vector database automatically if it does not exist
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    with st.spinner("Preparing medical knowledge database..."):
        build_index()

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1 style="font-size:42px; margin-bottom:5px;">
            🩺 AI Medical Assistant
        </h1>
        <p style="font-size:18px; color:#666;">
            Understand your medical reports with OCR, RAG and AI
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "ℹ️ Upload a medical report, review the extracted text, "
    "and let the AI explain the information in simple language."
)

st.caption(
    "Educational AI assistant for understanding medical reports "
    "using Retrieval-Augmented Generation."
)

st.warning(
    "⚠️ This application is for **educational purposes only** and does not provide "
    "medical diagnosis or replace professional medical advice. Please consult a "
    "qualified healthcare professional."
)


# ---------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "editable_report" not in st.session_state:
    st.session_state.editable_report = ""


# ---------------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------------

st.markdown("---")
st.subheader("📄 Step 1 — Provide Your Medical Report")
st.caption("Upload a report image or paste the report text manually.")

tab1, tab2 = st.tabs(
    ["📷 Upload Image", "⌨️ Type / Paste Text"]
)


# ---------------------------------------------------------------
# TAB 1: IMAGE UPLOAD
# ---------------------------------------------------------------

with tab1:

    uploaded_file = st.file_uploader(
        "Upload a JPG/JPEG/PNG image of a medical report",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Report",
            use_container_width=True
        )

        if st.button("🔍 Extract Text from Image", type="primary"):

            with st.spinner("Running OCR..."):

                extracted = extract_text_from_image(uploaded_file)

            # Store OCR result in BOTH session-state variables.
            st.session_state.report_text = extracted
            st.session_state.editable_report = extracted

            st.success(
                "Text extracted below. You can edit it before analyzing."
            )

            # Rerun so that the text area immediately displays
            # the newly extracted OCR text.
            st.rerun()


# ---------------------------------------------------------------
# TAB 2: TYPE / PASTE TEXT
# ---------------------------------------------------------------

with tab2:

    manual_text = st.text_area(
        "Paste or type your medical report here",

        placeholder=(
            "Hemoglobin: 10.2 g/dL\n"
            "WBC: 8,000 /µL\n"
            "Platelets: 250,000 /µL\n"
            "Glucose: 110 mg/dL"
        ),

        height=150
    )

    if manual_text.strip():

        st.session_state.report_text = manual_text
        st.session_state.editable_report = manual_text


# ---------------------------------------------------------------
# REVIEW / EDIT EXTRACTED TEXT
# ---------------------------------------------------------------

st.markdown("---")
st.subheader("✏️ Step 2 — Review & Edit Extracted Text")
st.caption(
    "Check the OCR output carefully and correct any mistakes before analysis."
)

edited_text = st.text_area(
    "Extracted / Editable Report Text",

    height=180,

    key="editable_report"
)


# Keep the edited version available for analysis.
st.session_state.report_text = edited_text


# ---------------------------------------------------------------
# RAG DEMONSTRATION OPTION
# ---------------------------------------------------------------

show_demo = st.checkbox(
    "Show RAG process details (for demonstration)",
    value=True
)


# ---------------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------------

analyze_clicked = st.button(
    "🔍 Analyze Report",
    type="primary"
)


# ---------------------------------------------------------------
# ANALYSIS / RAG PIPELINE
# ---------------------------------------------------------------

if analyze_clicked:

    if not edited_text.strip():

        st.error(
            "Please provide a report via image upload or manual text input first."
        )

    else:

        with st.spinner(
            "Retrieving relevant medical information and generating explanation..."
        ):

            result = run_rag_pipeline(edited_text)


        # -------------------------------------------------------
        # RESULTS
        # -------------------------------------------------------

        st.header("3. Results")

        st.subheader("🧾 AI Explanation")

        st.markdown(result["answer"])


        # -------------------------------------------------------
        # RAG DEMONSTRATION
        # -------------------------------------------------------

        if show_demo:

            st.divider()

            st.subheader(
                "🔬  How the AI Reached This Result")
            
            st.caption("This section shows the Retrieval-Augmented Generation process used by the application.")


            with st.expander("1. User Input (Query)"):

                st.code(result["query"])


            with st.expander( "📚 Retrieved Knowledge"):

                for chunk in result["retrieved_chunks"]:

                    st.markdown(
                        f"**Source: `{chunk['source']}`** "
                        f"(similarity score: {chunk['score']:.3f})"
                    )

                    st.text(chunk["text"])

                    st.markdown("---")


            with st.expander(
                "3. Retrieved Context (inserted into the prompt)"
            ):

                st.text(result["context"])


            with st.expander(
                "4. Full Prompt Sent to the LLM"
            ):

                st.text(result["prompt"])


            with st.expander(
                "5. Final LLM Answer (raw)"
            ):

                st.text(result["answer"])


        # -------------------------------------------------------
        # MEDICAL DISCLAIMER
        # -------------------------------------------------------

        st.divider()

        st.info(
            "**Medical Disclaimer:** This explanation was generated by an AI system "
            "for educational purposes only. It is not a medical diagnosis. Reference "
            "ranges vary between laboratories and individuals. Always consult a "
            "qualified healthcare professional about your actual health and test results."
        )
