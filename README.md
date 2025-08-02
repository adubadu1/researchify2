# Researchify2

A Streamlit web app that uses OpenAI to help you analyze your dataset. Great for nontechnical data scientists, students, or researchers looking to speed up project discovery.

## 🧠 How It Works

1. You upload a dataset or provide a link to one.
2. You type in a research question that can be answered using the dataset (e.g., *"How many flights were delayed in 2023?"*).
3. It searches Kaggle using those keywords.
4. You get a list of real, downloadable datasets that match your research idea.

## 🚀 Live Demo

**Coming soon** (or add your deployment link here if you host it on [Streamlit Cloud](https://streamlit.io/cloud) or [HuggingFace Spaces](https://huggingface.co/spaces))

## 🛠 Built With

- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- Python

## 📁 Folder Structure

main/
├── app.py # Streamlit frontend interface

├── researcher.py # Handles OpenAI API and dataset search logic

├── requirements.txt # Python dependencies (pip install -r requirements.txt)

├── .env # API keys (not included in Git)

└── README.md # Project documentation
