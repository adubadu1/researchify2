import streamlit as st
import os
import pandas as pd
import requests
from io import StringIO
from researcher import GenericLLMCodeExecutor

st.set_page_config(page_title="Agentic Data Researcher", layout="centered")
st.title("🔍 Agentic Data Researcher")

st.markdown("Upload a dataset or provide a dataset URL. Then, ask a question you'd like to answer from the data.")

url_input = st.text_input("📎 Dataset URL (CSV):")
uploaded_file = st.file_uploader("📁 Or upload a CSV file:", type="csv")
question = st.text_input("🧠 What is your research question?")

executor = GenericLLMCodeExecutor()

if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None
if "code" not in st.session_state:
    st.session_state.code = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None

def load_dataset():
    try:
        if url_input:
            if "kaggle.com/datasets" in url_input:
                st.info("Kaggle dataset link detected. Attempting to download using Kaggle API...")
                import subprocess
                import tempfile
                kaggle_url = url_input
                # Extract dataset slug from Kaggle URL
                import re
                match = re.search(r"kaggle.com/datasets/([^/]+/[^/?#]+)", kaggle_url)
                if not match:
                    st.error("Could not parse Kaggle dataset URL. Please provide a valid Kaggle dataset link.")
                    return None
                dataset_slug = match.group(1)
                temp_dir = tempfile.mkdtemp()
                try:
                    # Download dataset using Kaggle CLI
                    result = subprocess.run([
                        "kaggle", "datasets", "download", "-d", dataset_slug, "-p", temp_dir
                    ], capture_output=True, text=True)
                    if result.returncode != 0:
                        st.error(f"Kaggle download failed: {result.stderr}")
                        return None
                    # Find the downloaded file (usually a zip)
                    import os, zipfile
                    files = os.listdir(temp_dir)
                    zip_files = [f for f in files if f.endswith(".zip")]
                    if not zip_files:
                        st.error("No zip file found in Kaggle download.")
                        return None
                    zip_path = os.path.join(temp_dir, zip_files[0])
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    # Find all CSV files and select the largest one (main data)
                    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                    if not csv_files:
                        st.error("No CSV file found in Kaggle dataset zip.")
                        return None
                    # Select the largest CSV file by size
                    largest_csv = max(csv_files, key=lambda f: os.path.getsize(os.path.join(temp_dir, f)))
                    csv_path = os.path.join(temp_dir, largest_csv)
                    df = pd.read_csv(csv_path)
                    st.success(f"Kaggle dataset '{largest_csv}' loaded successfully!")
                    return df
                except Exception as e:
                    st.error(f"Failed to download or extract Kaggle dataset: {e}")
                    return None
            if not url_input.lower().endswith(".csv"):
                st.warning("URL must point directly to a .csv file (e.g., ending in .csv)")
                return None

            response = requests.get(url_input)
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if "text/csv" in content_type or url_input.endswith(".csv"):
                    df = pd.read_csv(StringIO(response.text))
                    st.success("Dataset loaded from URL successfully!")
                    return df
                else:
                    st.error("The URL did not return a valid CSV file.")
                    return None
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

if st.button("🚀 Run Analysis"):
    df = load_dataset()
    if df is not None and question.strip():
        st.session_state.df = df
        st.info(f"Columns detected in dataset: {', '.join(df.columns)}")
        with st.spinner("Generating answer from data..."):
            code = executor.generate_answer_code(df, question)
            st.session_state.code = code

            try:
                answer, error = executor.execute_code(code, df)
                st.session_state.result = answer

                if error:
                    st.error(f"Code execution error: {error}")
                elif answer is None:
                    st.warning("The model could not determine an answer to your question.")
                else:
                    st.success("Answer computed!")
                    explanation = executor.generate_explanation(question, answer, code, df)
                    st.session_state.explanation = explanation

                    st.subheader("✅ Answer:")
                    st.write(answer)

                    st.subheader("📊 How was this answer computed?")
                    st.markdown(explanation)

            except Exception as exec_error:
                st.error(f"Execution failed: {exec_error}")
    else:
        st.warning("Please make sure both a dataset and research question are provided.")

if st.session_state.code:
    with st.expander("🧾 Generated Code"):
        st.code(st.session_state.code, language="python")
