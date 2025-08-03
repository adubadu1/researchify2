# Researchify2

A goal-driven data science agent built with Streamlit and OpenAI. Designed for very nontechnical researchers, students, or anyone who wants to turn a research question into actionable analysis—no coding required.

## 🧠 How It Works

1. You type in a research question (e.g., *"How many flights were delayed in 2023?"*).
2. The agent automatically searches Kaggle for the most relevant dataset.
3. It downloads, loads, and analyzes the dataset using AI-generated code.
4. You get clear results and explanations—no technical steps required.

## 🚀 Live Demo

**Live on Streamlit Cloud:** [https://researchify2.streamlit.app/](https://researchify2.streamlit.app/)

## 🛠 Built With

- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://platform.openai.com/)
- Python
- Kaggle API (automatic dataset search and download)

## 📁 Folder Structure

main/
├── app.py            # Streamlit interface (agentic workflow: question → dataset → analysis)
├── researcher.py     # Handles OpenAI API, dataset search, and autonomous analysis
├── requirements.txt  # Python dependencies (pip install -r requirements.txt)
├── Dockerfile        # Containerization instructions for Docker
├── .env              # API keys (not included in Git)
└── README.md         # Project documentation
