import streamlit as st
import os
import pandas as pd
import requests
from io import StringIO
from researcher import GenericLLMCodeExecutor
from datasets import load_dataset as hf_load_dataset

st.set_page_config(page_title="Agentic Data Researcher", layout="centered")
st.title("🔍 Agentic Data Researcher")

st.markdown("Interact with the agentic researcher via chat. Upload a dataset or ask questions about the current dataset.")

executor = GenericLLMCodeExecutor()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "code" not in st.session_state:
    st.session_state.code = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
def add_chat(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})

def reset_context():
    st.session_state.df = None
    st.session_state.result = None
    st.session_state.code = None
    st.session_state.explanation = None
    st.session_state.dataset_name = None
    st.session_state.chat_history = []

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
    # Safety checks for dataset URL and uploaded files
    SUSPICIOUS_KEYWORDS = ["jailbreak", "prompt injection", "bypass", "ignore previous", "system: ", "act as", "simulate", "exploit", "malicious"]
    FORMULA_PREFIXES = ["=", "+", "-", "@"]
    MAX_ROWS = 100000
    MAX_COLS = 100
    MAX_FILE_SIZE_MB = 200
    def warn_if_suspicious(df, url=None):
        # Check for suspicious keywords in columns and cell values
        found = False
        for col in df.columns:
            for kw in SUSPICIOUS_KEYWORDS:
                if kw in str(col).lower():
                    found = True
        for col in df.columns:
            vals = df[col].astype(str).str.lower()
            for kw in SUSPICIOUS_KEYWORDS:
                if any(kw in v for v in vals):
                    found = True
        if url and any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS):
            found = True
        if found:
            st.warning("⚠️ This dataset may contain unsafe or offensive jailbreak/prompt injection content. Use caution when interpreting results.")
        # Formula injection check
        for col in df.columns:
            vals = df[col].astype(str)
            if any(any(vals.str.startswith(prefix)) for prefix in FORMULA_PREFIXES):
                st.warning("⚠️ This dataset may contain cells that could trigger formulas in spreadsheet software. Avoid exporting sensitive data.")
                break
        # Resource exhaustion check
        if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
            st.warning(f"⚠️ Large dataset detected ({df.shape[0]} rows, {df.shape[1]} columns). This may slow down or crash the app.")
        # Data type confusion
        for col in df.columns:
            if df[col].dtype == object:
                if df[col].apply(lambda x: isinstance(x, bytes) or '\x00' in str(x)).any():
                    st.warning(f"⚠️ Column '{col}' contains binary data or null bytes. This may cause errors.")
        # HTML/script injection
        import re
        html_pattern = re.compile(r'<(script|img|iframe|svg|object|embed|a|style|form|input|button|link|meta|base|applet|frame|frameset|marquee|video|audio|canvas|textarea|select|option|map|area|noscript|param|source|track|data|datalist|output|progress|meter|template|picture|portal|slot|summary|details|dialog|menu|menuitem|h1|h2|h3|h4|h5|h6|p|div|span|table|thead|tbody|tfoot|tr|th|td|ul|ol|li|dl|dt|dd|blockquote|pre|code|address|cite|b|i|u|em|strong|small|mark|del|ins|sub|sup|br|hr|wbr|abbr|acronym|bdo|big|center|dfn|kbd|q|rt|ruby|s|samp|tt|var|xmp)[^>]*>', re.IGNORECASE)
        for col in df.columns:
            if df[col].astype(str).apply(lambda x: bool(html_pattern.search(x))).any():
                st.warning(f"⚠️ Column '{col}' contains HTML or script tags. This may be unsafe to render.")
                break
        # Path traversal check (for filenames)
        if url and (".." in url or url.startswith("/") or url.startswith("~")):
            st.warning("⚠️ Suspicious file path detected in dataset URL. This may be unsafe.")
    try:
        if url_input:
            # Hugging Face dataset support
            if "huggingface.co/datasets/" in url_input:
                st.info("Hugging Face dataset link detected. Attempting to download using datasets library...")
                import re
                match = re.search(r"huggingface.co/datasets/([\w\-]+/[\w\-]+)", url_input)
                if not match:
                    st.error("Could not parse Hugging Face dataset URL. Please provide a valid link.")
                    return None
                hf_id = match.group(1)
                try:
                    ds = hf_load_dataset(hf_id)
                    split = max(ds.keys(), key=lambda k: len(ds[k]))
                    df = ds[split].to_pandas()
                    if df.shape[0] > 10000:
                        st.warning(f"⚠️ Dataset has {df.shape[0]} rows. Only the first 10,000 rows will be loaded to avoid memory issues.")
                        df = df.head(10000)
                    st.success(f"Hugging Face dataset '{hf_id}' loaded successfully!")
                    warn_if_suspicious(df, url_input)
                    return df
                except Exception as e:
                    st.error(f"Failed to download or load Hugging Face dataset: {e}")
                    return None
            # If user provides a Hugging Face dataset ID directly
            if url_input.startswith("hf:"):
                hf_id = url_input[3:]
                try:
                    ds = hf_load_dataset(hf_id)
                    split = max(ds.keys(), key=lambda k: len(ds[k]))
                    df = ds[split].to_pandas()
                    if df.shape[0] > 10000:
                        st.warning(f"⚠️ Dataset has {df.shape[0]} rows. Only the first 10,000 rows will be loaded to avoid memory issues.")
                        df = df.head(10000)
                    st.success(f"Hugging Face dataset '{hf_id}' loaded successfully!")
                    warn_if_suspicious(df, url_input)
                    return df
                except Exception as e:
                    st.error(f"Failed to download or load Hugging Face dataset: {e}")
                    return None
            # Kaggle dataset support
            if "kaggle.com/datasets" in url_input:
                st.info("Kaggle dataset link detected. Attempting to download using Kaggle API...")
                import subprocess
                import tempfile
                kaggle_url = url_input
                import re
                match = re.search(r"kaggle.com/datasets/([^/]+/[^/?#]+)", kaggle_url)
                if not match:
                    st.error("Could not parse Kaggle dataset URL. Please provide a valid Kaggle dataset link.")
                    return None
                dataset_slug = match.group(1)
                temp_dir = tempfile.mkdtemp()
                try:
                    result = subprocess.run([
                        "kaggle", "datasets", "download", "-d", dataset_slug, "-p", temp_dir
                    ], capture_output=True, text=True)
                    if result.returncode != 0:
                        st.error(f"Kaggle download failed: {result.stderr}")
                        return None
                    import os, zipfile
                    files = os.listdir(temp_dir)
                    zip_files = [f for f in files if f.endswith(".zip")]
                    if not zip_files:
                        st.error("No zip file found in Kaggle download.")
                        return None
                    zip_path = os.path.join(temp_dir, zip_files[0])
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                    if not csv_files:
                        st.error("No CSV file found in Kaggle dataset zip.")
                        return None
                    largest_csv = max(csv_files, key=lambda f: os.path.getsize(os.path.join(temp_dir, f)))
                    csv_path = os.path.join(temp_dir, largest_csv)
                    df = pd.read_csv(csv_path)
                    if df.shape[0] > 10000:
                        st.warning(f"⚠️ Dataset has {df.shape[0]} rows. Only the first 10,000 rows will be loaded to avoid memory issues.")
                        df = df.head(10000)
                    st.success(f"Kaggle dataset '{largest_csv}' loaded successfully!")
                    warn_if_suspicious(df, url_input)
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
                    warn_if_suspicious(df, url_input)
                    return df
                else:
                    st.error("The URL did not return a valid CSV file.")
                    return None
            else:
                st.error(f"Failed to download dataset from URL (status code {response.status_code})")
                return None

        elif uploaded_file:
            # Use uploaded_file.size for file size check (Streamlit provides this attribute)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.warning(f"⚠️ Uploaded file is too large ({file_size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.")
                return None
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Uploaded dataset loaded successfully!")
                warn_if_suspicious(df)
                return df
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
                return None
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
                    import pandas as pd
                    import numpy as np
                    if isinstance(answer, pd.DataFrame):
                        st.dataframe(answer)
                    elif isinstance(answer, (np.generic, np.ndarray)):
                        st.write(answer.item() if hasattr(answer, 'item') else answer.tolist())
                    else:
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
