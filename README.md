# Researchify 2
<<<<<<< HEAD
=======

>>>>>>> 0ae47b1 (Add note: agent currently supports only simple data science tasks)

An agentic web app built with Streamlit, OpenAI, and Kaggle to help nontechnical researchers analyze their datasets.

> **Note:** The agent currently performs only simple data science tasks (e.g., basic analysis, summary statistics, simple visualizations). Advanced analytics and complex modeling are not supported yet.

## 🧠 How It Works

1. You upload a dataset or provide a link (including Kaggle links).
2. You type in a research question that can be answered using the dataset (e.g., *"How many flights were delayed in 2023?"*).
3. If a Kaggle link is provided, the app automatically downloads the dataset from Kaggle.
4. The agent performs data science analysis to answer your question and provides results and explanations.

## 🚀 Live Demo

**Live on Streamlit Cloud:** [https://researchify2.streamlit.app/](https://researchify2.streamlit.app/)

## 🛠 Built With

- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- Python
- Kaggle API (automatic dataset search and download)

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
