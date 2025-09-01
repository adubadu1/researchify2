# Researchify 2

An agentic web app built with Streamlit, OpenAI, Kaggle and Hugging Face to help researchers analyze questions about their datasets. 

## How it works
The user inputs their dataset or a link to one, along with a question they want to answer about the dataset. Then the app generates the code in Python for the answer and displays the answer, along with an explanation of the answer.


## 🚀 Live Demo

**Live on Streamlit Cloud:** [https://researchify2-1.streamlit.app/](https://researchify2-1.streamlit.app/)

## 🛠 Built With

- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenAI API](https://platform.openai.com/) - GPT-4 code generation
- **Universal Package System** - Automatic installation of ANY PyPI library
- [Kaggle API](https://www.kaggle.com/docs/api) - Automatic dataset downloads
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) - ML dataset integration
- **Minimal Core Dependencies** - Lean requirements.txt with on-demand expansion


## 📁 Project Structure

```
researchify2/
├── app.py                    # Streamlit web interface with optimized backends
├── researcher.py             # AI analysis engine + Dynamic Globals System
├── requirements.txt          # Core dependencies (12 packages)
├── Dockerfile               # Container configuration with matplotlib support
├── .env.template            # Environment variables template
├── .dockerignore            # Docker build optimization
├── docker-compose.yml       # Container orchestration
├── run.sh                   # Quick start script
└── README.md               # This documentation
```

## 🐳 **Deployment**

### **Quick Start (Recommended):**
```bash
# 1. Clone repository
git clone https://github.com/adamsmohib/researchify2.git
cd researchify2

# 2. Setup environment
cp .env.template .env
# Edit .env with your API keys

# 3. Run with Docker (includes universal package system)
docker build -t researchify2 .
docker run -p 8501:8501 --env-file .env researchify2
```

### **Local Development:**
```bash
# Install core requirements
pip install -r requirements.txt

# Run locally (limited to pre-installed packages)
streamlit run app.py
```

## 🔑 **Configuration**

Create `.env` file with your API keys:
```bash
OPENAI_API_KEY="your_openai_api_key_here"
KAGGLE_USERNAME="your_kaggle_username_here"  
KAGGLE_KEY="your_kaggle_api_key_here"
```
