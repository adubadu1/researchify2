# Researchify 2


An agentic web app built with Streamlit, OpenAI, Kaggle, and Hugging Face to help nontechnical researchers analyze their datasets.

> **Note:** The agent currently performs only simple data science tasks (e.g., basic analysis, summary statistics, simple visualizations). Advanced analytics and complex modeling are not supported yet.

## 🧠 How It Works

1. You upload a dataset or provide a link (including Kaggle or Hugging Face links).
2. You type in a research question that can be answered using the dataset (e.g., *"How many flights were delayed in 2023?"*).
3. If a Kaggle or Hugging Face link is provided, the app automatically downloads the dataset using the appropriate API/library.
4. The agent performs data science analysis to answer your question and provides results and explanations.
5. **Safety checks** are performed on all uploaded and downloaded datasets to protect against prompt injection, formula injection, resource exhaustion, path traversal, HTML/script injection, and binary/null byte issues.

### Hugging Face Support

- Paste a Hugging Face dataset link (e.g., `https://huggingface.co/datasets/username/datasetname`) or ID (e.g., `hf:username/datasetname`) in the dataset URL field.
- The app will automatically download and load the dataset using the Hugging Face `datasets` library.
- Works for public datasets; private datasets may require authentication.

### Safety Features

- **Prompt Injection Protection:** Scans for suspicious keywords ("jailbreak", "prompt injection", "bypass", etc.) in columns, cell values, and URLs.
- **Formula Injection Protection:** Warns if any cell starts with `=`, `+`, `-`, or `@` (Excel formula risk).
- **Resource Exhaustion:** Limits maximum rows, columns, and file size for datasets.
- **Path Traversal:** Sanitizes filenames and blocks suspicious patterns in URLs.
- **HTML/Script Injection:** Warns if any cell contains HTML or script tags.
- **Data Type Confusion:** Warns if binary data or null bytes are detected in the dataset.

If any safety issue is detected, the app will display a warning and advise caution before proceeding with analysis.

## 🚀 Live Demo

**Live on Streamlit Cloud:** [https://researchify2.streamlit.app/](https://researchify2.streamlit.app/)

## 🛠 Built With

- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- Python
- Kaggle API (automatic dataset search and download)
- Hugging Face Datasets (automatic dataset download)

## 📁 Folder Structure

```
main/
├── app.py            # Streamlit interface (agentic workflow: question → dataset → analysis)
├── researcher.py     # Handles OpenAI API, dataset search, and autonomous analysis
├── requirements.txt  # Python dependencies (pip install -r requirements.txt)
├── Dockerfile        # Containerization instructions for Docker
├── .env              # API keys (not included in Git)
└── README.md         # Project documentation
```
