import streamlit as st
import os
import pandas as pd
import requests
from io import StringIO, BytesIO
from researcher import GenericLLMCodeExecutor

st.set_page_config(page_title="Agentic Data Researcher", layout="centered")
st.title("🔍 Agentic Data Researcher")

st.markdown("Upload a dataset or provide a dataset URL. Then, ask a question you'd like to answer from the data.")

# Option 1: URL input for remote dataset
url_input = st.text_input("📎 Dataset URL (CSV):")

# Option 2: Upload local CSV
uploaded_file = st.file_uploader("📁 Or upload a CSV file:", type="csv")

# Research question input
question = st.text_input("🧠 What is your research question?")

executor = GenericLLMCodeExecutor()

# Session states
if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None
if "code" not in st.session_state:
    st.session_state.code = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None

# Handle dataset loading
def load_dataset():
    try:
        if url_input:
            if "kaggle.com/datasets" in url_input:
                st.error("Direct Kaggle dataset links are not supported. Please download the CSV manually and upload it.")
                return None
            if not url_input.lower().endswith(".csv"):
                st.warning("URL must point directly to a .csv file (e.g., ending in .csv)")
                return None

            response = requests.get(url_input)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                st.success("Dataset loaded from URL successfully!")
                return df
            else:
                st.error(f"Failed to download dataset from URL (status code {response.status_code})")
                return None

        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Uploaded dataset loaded successfully!")
            return df
        else:
            st.warning("Please upload a CSV file or provide a dataset URL.")
            return None
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

# Analyze dataset
if st.button("🚀 Run Analysis"):
    df = load_dataset()
    if df is not None and question.strip():
        st.session_state.df = df
        with st.spinner("Generating answer from data..."):
            code = executor.generate_answer_code(df, question)
            st.session_state.code = code

            answer, error = executor.execute_code(code, df)
            st.session_state.result = answer

            if error:
                st.error(f"Code execution error: {error}")
            else:
                st.success("Answer computed!")
                explanation = executor.generate_explanation(question, answer, code)
                st.session_state.explanation = explanation
    else:
        st.warning("Please make sure both a dataset and research question are provided.")

# Display results
if st.session_state.result is not None:
    st.subheader("✅ Answer:")
    st.write(st.session_state.result)

if st.session_state.explanation:
    with st.expander("📊 How was this answer computed?"):
        st.markdown(st.session_state.explanation)
