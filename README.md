# Researchify 2

An agentic web app built with Streamlit, OpenAI, and Kaggle to help nontechnical researchers analyze their datasets.

## 🧠 How It Works

1. You upload a dataset or provide a link to one.
2. You type in a research question that can be answered using the dataset (e.g., *"How many flights were delayed in 2023?"*).
3. It searches Kaggle using those keywords.
4. You get a list of real, downloadable datasets that match your research idea.

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
