# Researchify 2

An enterprise-grade agentic web app built with Streamlit, OpenAI, Kaggle, and Hugging Face to help researchers analyze datasets with **universal automatic library installation** and **advanced signal processing capabilities**.

> **🚀 LATEST:** Revolutionary **Dynamic Globals System** with **FFT Analysis** - Ask any data science question including sophisticated signal processing and the system automatically provides required capabilities safely!

## 🌟 **Dynamic Globals System with Advanced Analytics**

Researchify 2 now features **enterprise-grade universal package management** with **sophisticated signal processing capabilities**:

### **🎯 Revolutionary Execution Engine:**
- **Dynamic Package Resolution** - Automatically provides required libraries during code execution
- **SmartGlobals System** - Intelligent namespace management with caching
- **FFT & Signal Processing** - Built-in support for frequency domain analysis
- **Statistical Computing** - Pre-loaded statsmodels, scipy.stats integration
- **Real-time PyPI Integration** - Validates and installs packages safely
- **Zero Configuration** - No need to update requirements.txt for new packages
- **Streamlined Architecture** - Simplified execution without import patching

### **🔬 Advanced Signal Processing Capabilities:**
- ✅ **FFT Analysis** - Built-in Fast Fourier Transform with proper frequency domain plotting
- ✅ **Frequency Domain Visualization** - Correct x-axis (frequencies) vs y-axis (amplitudes) plotting
- ✅ **Statistical Distributions** - CDF, percentile analysis with shading
- ✅ **Time Series Analysis** - ARIMA, seasonal decomposition, stationarity tests
- ✅ **Pre-loaded Scientific Functions** - numpy.fft, scipy.stats, statsmodels ready
- ✅ **Intelligent Type Conversion** - NumPy, Pandas, PyTorch types to native Python

### **🔒 Enterprise Security Features:**
- ✅ **400+ Trusted Package Whitelist** (pandas, sklearn, tensorflow, etc.)
- ✅ **Malicious Pattern Detection** (blocks packages with suspicious names)
- ✅ **PyPI Metadata Validation** (author verification, domain checking)
- ✅ **Installation Timeouts** (prevents resource exhaustion)
- ✅ **Homograph Attack Protection** (prevents typosquatting)
- ✅ **Dynamic Security Screening** (real-time package verification)

### **📊 Universal Package Categories:**
- 🔬 **Data Science:** pandas, numpy, scipy, statsmodels (pre-loaded)
- 📈 **Signal Processing:** FFT, frequency analysis, spectral methods (built-in)
- 🤖 **Machine Learning:** scikit-learn, tensorflow, torch, xgboost, lightgbm
- � **Visualization:** matplotlib, seaborn, plotly (optimized backends)
- 🖼️ **Computer Vision:** opencv, pillow, scikit-image, imageio
- 🌐 **Web & APIs:** requests, beautifulsoup4, scrapy, selenium
- 📝 **NLP & Text:** nltk, spacy, textblob, transformers, gensim
- 🗄️ **Databases:** sqlalchemy, pymongo, psycopg2, redis
- **And 400,000+ more packages safely!**

## 🧠 How It Works

1. **Upload or Link Dataset** - CSV files, Kaggle datasets, or Hugging Face datasets
2. **Ask Any Question** - From basic stats to advanced ML: *"Build a neural network to predict stock prices"*
3. **Automatic Package Resolution** - System detects required libraries and installs them safely
4. **AI Code Generation** - GPT-4 generates appropriate analysis code  
5. **Secure Execution** - Code runs in sandboxed environment with installed packages
6. **Instant Results** - Get analysis, visualizations, and insights immediately

### 🔧 **Advanced Dynamic Execution Engine**

The revolutionary execution system now handles sophisticated data science workflows:

- **🎯 Smart Import Resolution:** Automatic package detection and installation
- **🔄 Dynamic Globals System:** SmartGlobals with caching and namespace management
- **⚡ Pre-loaded Scientific Computing:** FFT, stats, visualization ready instantly
- **🛡️ Security-First Architecture:** Multi-layer validation without compromising functionality
- **📚 Universal Library Access:** 400,000+ PyPI packages with enterprise protection
- **🔒 Simplified Execution:** Removed complex import patching for stability
- **🎨 Optimized Backends:** Matplotlib 'Agg' backend for containerized environments

### **💡 Advanced Analysis Examples Now Possible:**

```python
# Signal Processing & FFT Analysis
"Plot the FFT of this stock and shade the 90th percentile CDF"  # → Built-in FFT with proper frequency domain
"Analyze frequency components with spectral density"            # → Pre-loaded scipy.signal functions
"Perform wavelet transformation on time series"                # → Auto-installs PyWavelets

# Advanced Statistical Analysis  
"Run ARIMA forecasting with seasonal decomposition"            # → Pre-loaded statsmodels functions
"Perform Augmented Dickey-Fuller stationarity test"           # → Built-in adfuller, kpss tests
"Calculate rolling statistics with confidence intervals"        # → Integrated pandas + scipy.stats

# Deep Learning & AI
"Build a transformer model for text classification"            # → Auto-installs transformers, torch
"Create a CNN for image recognition with transfer learning"    # → Auto-installs tensorflow, keras
"Perform sentiment analysis with BERT embeddings"             # → Auto-installs transformers, datasets

# Advanced Visualization  
"Create interactive 3D scatter plots with animations"          # → Auto-installs plotly, matplotlib
"Build network graphs with community detection"               # → Auto-installs networkx, community
"Generate publication-ready statistical plots"                # → Pre-loaded seaborn, matplotlib

# Specialized Financial Analytics
"Calculate technical indicators and Bollinger bands"          # → Auto-installs TA-Lib, yfinance
"Perform Monte Carlo simulations for portfolio risk"         # → Built-in numpy.random, scipy.stats
"Analyze options pricing with Black-Scholes model"           # → Pre-loaded mathematical functions
```

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

- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenAI API](https://platform.openai.com/) - GPT-4 code generation
- **Universal Package System** - Automatic installation of ANY Python library
- [Kaggle API](https://www.kaggle.com/docs/api) - Automatic dataset downloads
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) - ML dataset integration
- **Minimal Core Dependencies** - Lean requirements.txt with on-demand expansion

## 🏗️ **Revolutionary Architecture**

### **Minimal Requirements + Dynamic Expansion Strategy**
Unlike traditional applications that pre-install hundreds of packages, Researchify 2 uses a **lean core + smart expansion** approach:

```txt
Core Requirements (requirements.txt):
├── streamlit              # Web framework
├── openai                # AI integration  
├── pandas & numpy        # Data fundamentals
├── matplotlib            # Visualization core
├── statsmodels           # Statistical analysis
├── python-dotenv         # Configuration
└── kaggle & datasets     # Data sources

Dynamic SmartGlobals System (400,000+ packages available):
├── Pre-loaded Functions:
│   ├── numpy.fft.*       # FFT analysis ready
│   ├── scipy.stats.*     # Statistical functions
│   ├── matplotlib.pyplot # Optimized plotting
│   └── statsmodels.*     # Time series analysis
├── On-demand Installation:
│   ├── scikit-learn      # ML workflows
│   ├── tensorflow/torch  # Deep learning
│   ├── plotly/seaborn    # Advanced visualization
│   └── ANY PyPI package  # Universal support
```

### **Enhanced Security Architecture**
- **🛡️ Multi-Layer Validation:** Package name analysis, PyPI verification, metadata checking
- **⚡ Real-time Screening:** Live validation against malicious package databases
- **🔒 Sandboxed Execution:** Isolated environment prevents system compromise
- **⏱️ Timeout Protection:** Installation limits prevent resource exhaustion
- **📋 Audit Logging:** Complete package installation history tracking
- **🎯 Smart Globals:** Pre-vetted namespace with secure dynamic expansion

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

> **Note:** Docker deployment strongly recommended for full universal package support with advanced security features and optimized backends.

## 🔑 **Configuration**

Create `.env` file with your API keys:
```bash
OPENAI_API_KEY="your_openai_api_key_here"
KAGGLE_USERNAME="your_kaggle_username_here"  
KAGGLE_KEY="your_kaggle_api_key_here"
```

## 🌟 **Why Researchify 2?**

- **🚀 Zero Setup:** No complex environment management required
- **🔒 Enterprise Security:** Production-ready package validation system
- **⚡ Lightning Fast:** Packages install in seconds with smart caching
- **🌍 Universal Coverage:** Access to entire Python ecosystem safely
- **🎯 AI-Powered:** GPT-4 generates optimal code with FFT guidelines
- **📊 Research-Grade:** From basic stats to cutting-edge signal processing
- **🔬 Pre-loaded Analytics:** FFT, statistics, and visualization ready instantly
- **🛡️ Simplified Architecture:** Stable execution without complex import patching

## � **No Limits Data Science**

With the absolute wildcard system, you can now perform **any data science task** without worrying about dependencies:

### **🎯 Advanced Analytics Examples:**
- **"Perform LSTM time series forecasting"** → Auto-installs: tensorflow, keras, sklearn
- **"Create a Dash web dashboard"** → Auto-installs: dash, plotly, flask  
- **"Build a recommendation system"** → Auto-installs: surprise, implicit, lightfm
- **"Analyze social networks"** → Auto-installs: networkx, community, igraph
- **"Process natural language"** → Auto-installs: spacy, nltk, transformers
- **"Perform Bayesian analysis"** → Auto-installs: pymc, arviz, bambi
- **"Create interactive maps"** → Auto-installs: folium, geopandas, leaflet

### **🔬 Research-Grade Capabilities:**
- **Bioinformatics:** biopython, scanpy, anndata
- **Econometrics:** linearmodels, arch, pyeconometrics  
- **Signal Processing:** scipy.signal, librosa, pyaudio
- **Optimization:** cvxpy, pulp, gekko
- **Quantum Computing:** qiskit, cirq, pennylane
- **Astronomy:** astropy, photutils, sunpy

### **🛡️ Security Guarantees:**
```
✅ Trusted Package Whitelist (400+ packages)
✅ Real-time PyPI Verification  
✅ Malicious Pattern Detection
✅ Domain Reputation Checking
✅ Installation Timeout Protection
✅ Metadata Security Scanning
```

**The result:** Ask any data science question and get enterprise-grade analysis without limits!

---

## 🔬 **Advanced Signal Processing & Analytics**

With the enhanced dynamic globals system, you can now perform **sophisticated signal processing and statistical analysis** without dependency concerns:

### **🎯 Signal Processing & FFT Examples:**
- **"Plot the FFT of this stock and shade the 90th percentile CDF"** → Built-in FFT with proper frequency domain visualization
- **"Analyze frequency components and identify dominant frequencies"** → Pre-loaded numpy.fft with fftfreq support
- **"Perform spectral analysis with power spectral density"** → Integrated scipy.signal functions
- **"Apply bandpass filtering to remove noise"** → Auto-installs advanced signal processing libraries
- **"Calculate cross-correlation between time series"** → Built-in statistical functions ready

### **📈 Advanced Statistical Analysis:**
- **"Run ARIMA forecasting with residual analysis"** → Pre-loaded statsmodels.tsa functions
- **"Perform Augmented Dickey-Fuller stationarity tests"** → Built-in adfuller function
- **"Calculate rolling statistics with confidence bands"** → Integrated pandas + scipy.stats
- **"Analyze seasonal decomposition patterns"** → Pre-loaded seasonal_decompose
- **"Compute correlation matrices with significance testing"** → Enhanced statistical capabilities

### **🤖 Enhanced Machine Learning & AI:**
- **"Build LSTM networks for time series prediction"** → Auto-installs: tensorflow, keras
- **"Create transformer models for NLP tasks"** → Auto-installs: transformers, torch
- **"Implement reinforcement learning algorithms"** → Auto-installs: gym, stable-baselines3
- **"Perform computer vision with pre-trained models"** → Auto-installs: opencv, torchvision
- **"Build recommendation systems with collaborative filtering"** → Auto-installs: surprise, implicit

### **📊 Publication-Ready Visualization:**
- **"Create interactive plotly dashboards with real-time updates"** → Auto-installs: plotly, dash
- **"Generate publication-ready matplotlib figures"** → Pre-loaded with optimized backends
- **"Build 3D scatter plots with animations"** → Auto-installs: plotly, matplotlib
- **"Create network graphs with community detection"** → Auto-installs: networkx, community
- **"Design geographic visualizations with folium"** → Auto-installs: folium, geopandas

### **🛡️ Enhanced Security & Stability:**
```
✅ 400+ Trusted Package Whitelist (expanded coverage)
✅ Real-time PyPI Verification with Metadata Checking
✅ Malicious Pattern Detection & Homograph Protection
✅ Domain Reputation & Author Verification
✅ Installation Timeout & Resource Protection
✅ Dynamic Security Screening & Audit Logging
✅ Pre-vetted Scientific Computing Environment
✅ Simplified Architecture for Maximum Stability
✅ No Import Patching Issues - Pure Dynamic Globals
✅ FFT Guidelines for Proper Frequency Domain Analysis
```

**The enhanced result:** Ask any data science or signal processing question and get enterprise-grade analysis with cutting-edge capabilities and rock-solid stability!
