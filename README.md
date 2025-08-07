# Researchify 2

An agentic web app built with Streamlit, OpenAI, Kaggle, and Hugging Face to help nontechnical researchers analyze their datasets with **automatic library installation**.

> **Note:** The agent currently performs only simple data science tasks (e.g., basic analysis, summary statistics, simple visualizations). Advanced analytics and complex modeling are not supported yet.

## 🚀 **NEW: Dynamic Package Installation**

Researchify 2 now features **intelligent auto-installation** of Python packages! When the AI generates code that requires additional libraries, the system automatically:

- **Detects** missing packages (sklearn, scipy, matplotlib, etc.)
- **Installs** them on-the-fly using pip
- **Maps** tricky package names (e.g., `sklearn` → `scikit-learn`)
- **Continues** execution seamlessly

### Supported Auto-Installation:
- ✅ **Machine Learning:** `sklearn` (scikit-learn), `xgboost`, `lightgbm`
- ✅ **Statistics:** `statsmodels`, `scipy`, statistical tests (adfuller, kpss)
- ✅ **Visualization:** `matplotlib`, `seaborn`, `plotly`
- ✅ **Computer Vision:** `opencv-python`, `Pillow`, `scikit-image`
- ✅ **And many more common packages...**

No more "ModuleNotFoundError" - the system handles dependencies automatically!

## 🧠 How It Works

1. You upload a dataset or provide a link (including Kaggle or Hugging Face links).
2. You type in a research question that can be answered using the dataset (e.g., *"How many flights were delayed in 2023?"*, *"Is this stock price stationary?"*, *"Predict next week's stock price"*).
3. If a Kaggle or Hugging Face link is provided, the app automatically downloads the dataset using the appropriate API/library.
4. The agent performs data science analysis to answer your question and provides results and explanations.
5. **🆕 Auto-Installation:** If the generated code requires additional packages (sklearn, scipy, etc.), they are automatically installed.
6. **Safety checks** are performed on all uploaded and downloaded datasets to protect against prompt injection, formula injection, resource exhaustion, path traversal, HTML/script injection, and binary/null byte issues.

### 🔧 Intelligent Code Execution

The system now features advanced code execution capabilities:

- **Dynamic Import Handling:** Automatically resolves `import sklearn`, `from statsmodels import`, etc.
- **Package Mapping:** Smart mapping of import names to pip packages (e.g., `cv2` → `opencv-python`)
- **Error Recovery:** If a package is missing, it gets installed and execution continues
- **Pre-loaded Libraries:** Common statistical and ML functions are pre-available
- **Dependency Resolution:** Handles complex package dependencies automatically

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
- Python with **Dynamic Package Installation**
- Kaggle API (automatic dataset search and download)
- Hugging Face Datasets (automatic dataset download)
- **Auto-installing libraries:** scikit-learn, statsmodels, scipy, matplotlib, seaborn, plotly, opencv, and more

## 🔧 Technical Features

### Dynamic Package Management
- **Smart Import Detection:** Automatically detects and installs missing packages during code execution
- **Package Mapping:** Handles tricky package names (sklearn → scikit-learn, cv2 → opencv-python, etc.)
- **Pre-loaded Libraries:** Common statistical functions (adfuller, kpss, seasonal_decompose) ready to use
- **Error Recovery:** Seamless installation and retry on import failures
- **Docker Compatible:** Works within containerized environments

### Code Execution Engine
- **Sandboxed Execution:** Safe code execution with controlled global environment
- **Import Patching:** Intelligently patches import statements for auto-installation
- **Library Caching:** Once installed, packages are cached for faster subsequent use
- **Fallback Handling:** Graceful degradation if package installation fails

## 📁 Folder Structure

```
main/
├── app.py            # Streamlit interface (agentic workflow: question → dataset → analysis)
├── researcher.py     # Handles OpenAI API, dataset search, autonomous analysis, and dynamic package installation
├── requirements.txt  # Core Python dependencies (additional packages auto-installed as needed)
├── Dockerfile        # Containerization instructions for Docker
├── .env              # API keys (not included in Git)
└── README.md         # Project documentation
```

## 🐳 Docker Deployment

The app includes Docker support with the new dynamic package installation feature:

```bash
# Build the Docker image
docker build -t researchify2 .

# Run the container
docker run -p 8501:8501 --env-file .env researchify2
```

The Docker container will automatically install additional packages as needed during runtime, making it highly flexible for various data science tasks.

## 📊 Example Use Cases

With the new auto-installation feature, you can now ask complex questions that require specialized libraries:

- **Time Series Analysis:** "Is this stock stationary?" (auto-installs statsmodels)
- **Machine Learning:** "Predict next week's stock price" (auto-installs scikit-learn)
- **Statistical Testing:** "Perform correlation analysis" (auto-installs scipy)
- **Advanced Visualization:** "Create interactive plots" (auto-installs plotly)
- **Computer Vision:** "Analyze image data" (auto-installs opencv, Pillow)

The system handles all the technical complexity behind the scenes!
