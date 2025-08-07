import os
import re
import json
import traceback
import subprocess
import sys
import hashlib
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from thefuzz import process

load_dotenv()

def convert_numpy_types(obj):
    """
    Convert numpy and other library types to native Python types for cleaner display.
    Handles scalars, arrays, lists, dicts, and nested structures from multiple libraries.
    """
    import numpy as np
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size == 1:
            # Single element array - convert to scalar
            return convert_numpy_types(obj.item())
        else:
            # Multi-element array - convert to list
            return obj.tolist()
    
    # Handle pandas types
    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj  # Keep DataFrames as-is for display
        elif hasattr(pd, 'NA') and obj is pd.NA:
            return None
        elif hasattr(obj, 'dtype') and hasattr(obj.dtype, 'name'):
            # Handle pandas scalar types
            if 'int' in obj.dtype.name:
                return int(obj)
            elif 'float' in obj.dtype.name:
                return float(obj)
            elif 'bool' in obj.dtype.name:
                return bool(obj)
    except ImportError:
        pass
    
    # Handle PyTorch tensors
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.tolist()
    except ImportError:
        pass
    
    # Handle TensorFlow tensors
    try:
        import tensorflow as tf
        if isinstance(obj, tf.Tensor):
            if obj.shape == ():
                return obj.numpy().item()
            else:
                return obj.numpy().tolist()
        elif hasattr(tf, 'EagerTensor') and isinstance(obj, tf.EagerTensor):
            if obj.shape == ():
                return obj.numpy().item()
            else:
                return obj.numpy().tolist()
    except ImportError:
        pass
    
    # Handle decimal.Decimal
    try:
        import decimal
        if isinstance(obj, decimal.Decimal):
            return float(obj)
    except ImportError:
        pass
    
    # Handle fractions.Fraction
    try:
        import fractions
        if isinstance(obj, fractions.Fraction):
            return float(obj)
    except ImportError:
        pass
    
    # Handle collections recursively
    if isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, set):
        return {convert_numpy_types(item) for item in obj}
    
    # Generic fallback for any object with conversion methods
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return obj.item()
        except:
            pass
    elif hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
        try:
            return obj.tolist()
        except:
            pass
    elif hasattr(obj, 'numpy') and callable(getattr(obj, 'numpy')):
        try:
            numpy_obj = obj.numpy()
            return convert_numpy_types(numpy_obj)
        except:
            pass
    
    # Return unchanged if not a convertible type
    return obj

# Security lists for package validation
MALICIOUS_PATTERNS = [
    'malware', 'virus', 'trojan', 'backdoor', 'keylogger', 'spyware',
    'hack', 'exploit', 'payload', 'shell', 'reverse', 'bind',
    'cryptominer', 'miner', 'stealer', 'grabber', 'rat',
    'botnet', 'ddos', 'bruteforce', 'crack', 'bypass'
]

SUSPICIOUS_PACKAGES = [
    'os-system', 'subprocess-shell', 'eval-exec', 'compile-exec',
    'file-write', 'network-request', 'socket-connect', 'urllib-open',
    'requests-get', 'curl-command', 'wget-download'
]

WHITELIST_DOMAINS = [
    'pypi.org', 'python.org', 'github.com', 'gitlab.com', 'bitbucket.org',
    'anaconda.org', 'conda-forge.org', 'pytorch.org', 'tensorflow.org',
    'scikit-learn.org', 'scipy.org', 'numpy.org', 'pandas.pydata.org',
    'matplotlib.org', 'plotly.com', 'seaborn.pydata.org'
]

TRUSTED_PACKAGES = {
    # Data Science & Analytics
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
    'statsmodels', 'scikit-learn', 'sklearn', 'xgboost', 'lightgbm',
    'catboost', 'optuna', 'hyperopt', 'shap', 'lime',
    
    # Machine Learning & AI
    'tensorflow', 'torch', 'pytorch', 'keras', 'transformers',
    'datasets', 'tokenizers', 'accelerate', 'diffusers',
    'opencv-python', 'opencv-contrib-python', 'pillow', 'imageio',
    'scikit-image', 'albumentations',
    
    # Web & APIs
    'requests', 'urllib3', 'httpx', 'aiohttp', 'flask', 'fastapi',
    'streamlit', 'dash', 'gradio', 'beautifulsoup4', 'lxml',
    
    # Utilities
    'tqdm', 'click', 'typer', 'rich', 'colorama', 'tabulate',
    'openpyxl', 'xlrd', 'xlsxwriter', 'pyyaml', 'toml', 'configparser',
    'python-dotenv', 'pathlib', 'datetime', 'dateutil',
    
    # Database & Storage
    'sqlalchemy', 'pymongo', 'redis', 'psycopg2', 'mysql-connector-python',
    's3fs', 'boto3', 'azure-storage-blob', 'google-cloud-storage',
    
    # Jupyter & Notebooks
    'jupyter', 'ipython', 'ipykernel', 'ipywidgets', 'nbformat',
    
    # Scientific Computing
    'sympy', 'networkx', 'igraph', 'graph-tool', 'pymc',
    'arviz', 'emcee', 'corner', 'astropy', 'biopython'
}

def is_package_safe(package_name):
    """
    Comprehensive security check for package safety.
    """
    package_lower = package_name.lower()
    
    # Check against malicious patterns
    for pattern in MALICIOUS_PATTERNS:
        if pattern in package_lower:
            return False, f"Package name contains suspicious pattern: {pattern}"
    
    # Check against suspicious package names
    for suspicious in SUSPICIOUS_PACKAGES:
        if suspicious in package_lower:
            return False, f"Package name matches suspicious pattern: {suspicious}"
    
    # Check if package is in trusted list
    if package_name in TRUSTED_PACKAGES:
        return True, "Package is in trusted whitelist"
    
    # Additional safety checks
    if len(package_name) < 2:
        return False, "Package name too short"
    
    if package_name.startswith('_') or package_name.startswith('.'):
        return False, "Package name starts with suspicious character"
    
    # Check for homograph attacks (similar looking characters)
    suspicious_chars = ['0', '1', 'l', 'I', 'O']
    if any(char in package_name for char in suspicious_chars):
        # Additional verification needed for packages with confusing characters
        pass
    
    return True, "Package appears safe"

def verify_package_on_pypi(package_name):
    """
    Verify package exists on PyPI and check its metadata for safety.
    """
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
        if response.status_code != 200:
            return False, "Package not found on PyPI"
        
        data = response.json()
        info = data.get('info', {})
        
        # Check package metadata
        author = info.get('author', '').lower()
        description = info.get('summary', '').lower()
        home_page = info.get('home_page', '').lower()
        
        # Check for suspicious content in metadata
        for pattern in MALICIOUS_PATTERNS:
            if pattern in description or pattern in author:
                return False, f"Suspicious content in package metadata: {pattern}"
        
        # Check if home page is from trusted domain
        if home_page:
            domain_trusted = any(domain in home_page for domain in WHITELIST_DOMAINS)
            if not domain_trusted and 'github.com' not in home_page:
                return False, "Package from untrusted domain"
        
        # Check download count (packages with very low downloads might be suspicious)
        downloads = data.get('urls', [])
        if len(downloads) == 0:
            return False, "No download URLs available"
        
        return True, "Package verified on PyPI"
        
    except requests.RequestException:
        return False, "Could not verify package on PyPI"
    except Exception as e:
        return False, f"Error verifying package: {str(e)}"

def safe_auto_install_package(package_name):
    """
    Safely install a package with comprehensive security checks.
    """
    # First check if already installed
    try:
        __import__(package_name)
        return True, "Package already installed"
    except ImportError:
        pass
    
    # Security checks
    is_safe, safety_reason = is_package_safe(package_name)
    if not is_safe:
        print(f"🚫 Security Alert: Blocked package '{package_name}' - {safety_reason}")
        return False, f"Security check failed: {safety_reason}"
    
    # Verify on PyPI (with timeout)
    is_verified, verify_reason = verify_package_on_pypi(package_name)
    if not is_verified:
        print(f"⚠️  Warning: Could not verify package '{package_name}' - {verify_reason}")
        # For trusted packages, still allow installation even if verification fails
        if package_name not in TRUSTED_PACKAGES:
            return False, f"Verification failed: {verify_reason}"
    
    # Proceed with installation
    try:
        print(f"✅ Installing safe package: {package_name}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name,
            "--no-warn-script-location", "--quiet"
        ], timeout=60)  # 60 second timeout
        
        # Verify installation worked
        try:
            __import__(package_name)
            print(f"✅ Successfully installed and verified: {package_name}")
            return True, "Package installed successfully"
        except ImportError:
            return False, "Package installed but could not import"
            
    except subprocess.TimeoutExpired:
        return False, "Installation timeout (package may be too large or slow)"
    except subprocess.CalledProcessError as e:
        return False, f"Installation failed: {e}"
    except Exception as e:
        return False, f"Unexpected error during installation: {e}"

def patch_imports_in_code(code):
    """
    Patch import statements in code to handle package installation with absolute wildcard support.
    """
    lines = code.split('\n')
    patched_lines = []
    
    # Known package mappings for tricky cases
    package_mappings = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'bs4': 'beautifulsoup4',
        'yaml': 'PyYAML'
    }
    
    import re
    
    for line in lines:
        original_line = line
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            patched_lines.append(original_line)
            continue
            
        # Check for any import statement
        import_match = re.match(r'^(from\s+(\w+)|import\s+(\w+))', stripped)
        if import_match:
            # Extract the package name
            package_name = None
            if import_match.group(2):  # from package import ...
                package_name = import_match.group(2)
            elif import_match.group(3):  # import package
                package_name = import_match.group(3)
            
            if package_name:
                # Skip built-in modules
                builtin_modules = {
                    'os', 'sys', 'json', 're', 'datetime', 'time', 'math', 'random',
                    'collections', 'itertools', 'functools', 'operator', 'copy',
                    'pickle', 'csv', 'io', 'pathlib', 'glob', 'shutil', 'tempfile',
                    'urllib', 'http', 'email', 'html', 'xml', 'sqlite3', 'hashlib',
                    'uuid', 'logging', 'warnings', 'traceback', 'inspect'
                }
                
                if package_name in builtin_modules:
                    patched_lines.append(original_line)
                else:
                    # Use safe auto-installation for ANY package
                    pip_package = package_mappings.get(package_name, package_name)
                    patched_lines.append(f"""try:
    {original_line}
except ImportError:
    import subprocess
    import sys
    # Safe installation with security checks
    try:
        from researcher import safe_auto_install_package
        success, message = safe_auto_install_package("{pip_package}")
        if success:
            {original_line}
        else:
            print(f"⚠️  Could not safely install {pip_package}: {{message}}")
            raise ImportError(f"Safe installation failed for {pip_package}")
    except Exception as e:
        print(f"🚫 Security check failed for package {pip_package}")
        raise ImportError(f"Package {pip_package} blocked for security reasons")""")
            else:
                patched_lines.append(original_line)
        else:
            patched_lines.append(original_line)
    
    return '\n'.join(patched_lines)

def smart_import(module_name, package_name=None):
    """
    Dynamically import a module, installing the package if needed.
    """
    if package_name is None:
        package_name = module_name
    
    try:
        return __import__(module_name)
    except ImportError:
        success, message = safe_auto_install_package(package_name)
        if success:
            try:
                return __import__(module_name)
            except ImportError:
                return None
        return None

def create_dynamic_globals():
    """
    Create a globals dictionary with dynamic importing capabilities.
    """
    # Common package mappings
    package_mappings = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'scipy': 'scipy'
    }
    
    # Pre-install common packages that might be needed
    def ensure_package_available(module_name, package_name=None):
        if package_name is None:
            package_name = package_mappings.get(module_name, module_name)
        
        try:
            return __import__(module_name)
        except ImportError:
            success, message = safe_auto_install_package(package_name)
            if success:
                try:
                    return __import__(module_name)
                except ImportError:
                    return None
            return None
    
    # Base globals with pandas and numpy
    globals_dict = {
        "pd": pd,
        "np": np,
        "__builtins__": __builtins__,
        # Add common numpy functions to prevent confusion
        "abs": abs,
        "len": len,
        "max": max,
        "min": min,
        "sum": sum,
        "round": round,
        # Add essential Python built-ins
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AttributeError": AttributeError,
        "ImportError": ImportError,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }
    
    # Pre-load common numpy.fft functions
    try:
        globals_dict.update({
            "fft": np.fft.fft,
            "fftfreq": np.fft.fftfreq,
            "ifft": np.fft.ifft,
            "fftshift": np.fft.fftshift,
        })
    except ImportError:
        pass
    
    # Pre-load common statistical functions
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.tsa.seasonal import seasonal_decompose
        globals_dict.update({
            "adfuller": adfuller,
            "kpss": kpss,
            "seasonal_decompose": seasonal_decompose
        })
    except ImportError:
        # These will be handled by the dynamic importer
        pass
    
    try:
        import scipy.stats as stats
        globals_dict["stats"] = stats
    except ImportError:
        pass
    
    # Pre-load common visualization libraries
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        globals_dict["plt"] = plt
        globals_dict["matplotlib"] = matplotlib
    except ImportError:
        pass
    
    try:
        import seaborn as sns
        globals_dict["sns"] = sns
        globals_dict["seaborn"] = sns
    except ImportError:
        pass
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly
        globals_dict["px"] = px
        globals_dict["go"] = go
        globals_dict["plotly"] = plotly
    except ImportError:
        pass
    
    # Pre-load sklearn if available or install it
    sklearn_module = ensure_package_available('sklearn', 'scikit-learn')
    if sklearn_module:
        globals_dict["sklearn"] = sklearn_module
        # Also add common sklearn submodules
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            globals_dict.update({
                "train_test_split": train_test_split,
                "LinearRegression": LinearRegression,
                "RandomForestRegressor": RandomForestRegressor,
                "mean_squared_error": mean_squared_error,
                "r2_score": r2_score
            })
        except ImportError:
            pass
    
    # Pre-load datetime handling
    try:
        import datetime
        globals_dict["datetime"] = datetime
        globals_dict["timedelta"] = datetime.timedelta
    except ImportError:
        pass
    
    class DynamicImporter:
        def __getitem__(self, name):
            # Handle special statistical functions
            if name == 'adfuller':
                success, _ = safe_auto_install_package('statsmodels')
                if success:
                    from statsmodels.tsa.stattools import adfuller
                    return adfuller
                raise ImportError(f"Cannot import {name}")
            elif name == 'kpss':
                success, _ = safe_auto_install_package('statsmodels')
                if success:
                    from statsmodels.tsa.stattools import kpss
                    return kpss
                raise ImportError(f"Cannot import {name}")
            elif name == 'seasonal_decompose':
                success, _ = safe_auto_install_package('statsmodels')
                if success:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    return seasonal_decompose
                raise ImportError(f"Cannot import {name}")
            elif name == 'stats':
                success, _ = safe_auto_install_package('scipy')
                if success:
                    import scipy.stats as stats
                    return stats
                raise ImportError(f"Cannot import {name}")
            elif name == 'fft':
                # Handle FFT function directly from numpy
                try:
                    return np.fft.fft  # Return the actual function, not the module
                except ImportError:
                    raise ImportError(f"Cannot import {name}")
            elif name == 'fftfreq':
                # Handle fftfreq function from numpy
                try:
                    return np.fft.fftfreq  # Return the actual function, not the module
                except ImportError:
                    raise ImportError(f"Cannot import {name}")
            elif name == 'ifft':
                # Handle inverse FFT function from numpy
                try:
                    return np.fft.ifft  # Return the actual function, not the module
                except ImportError:
                    raise ImportError(f"Cannot import {name}")
            elif name == 'fftshift':
                # Handle fftshift function from numpy
                try:
                    return np.fft.fftshift
                except ImportError:
                    raise ImportError(f"Cannot import {name}")
            elif name == 'abs':
                # Make sure abs function is available
                return abs
            elif name == 'len':
                # Make sure len function is available
                return len
            elif name == 'sklearn':
                success, _ = safe_auto_install_package('scikit-learn')
                if success:
                    import sklearn
                    return sklearn
                raise ImportError(f"Cannot import {name}")
            elif name == 'plt':
                success, _ = safe_auto_install_package('matplotlib')
                if success:
                    import matplotlib.pyplot as plt
                    return plt
                raise ImportError(f"Cannot import {name}")
            elif name == 'matplotlib':
                success, _ = safe_auto_install_package('matplotlib')
                if success:
                    import matplotlib
                    return matplotlib
                raise ImportError(f"Cannot import {name}")
            elif name == 'sns' or name == 'seaborn':
                success, _ = safe_auto_install_package('seaborn')
                if success:
                    import seaborn as sns
                    return sns
                raise ImportError(f"Cannot import {name}")
            elif name == 'plotly':
                success, _ = safe_auto_install_package('plotly')
                if success:
                    import plotly
                    return plotly
                raise ImportError(f"Cannot import {name}")
            elif name == 'px':
                success, _ = safe_auto_install_package('plotly')
                if success:
                    import plotly.express as px
                    return px
                raise ImportError(f"Cannot import {name}")
            elif name == 'go':
                success, _ = safe_auto_install_package('plotly')
                if success:
                    import plotly.graph_objects as go
                    return go
                raise ImportError(f"Cannot import {name}")
            elif name == 'datetime':
                # Handle datetime module
                import datetime
                return datetime
            elif name == 'timedelta':
                # Handle timedelta class
                import datetime
                return datetime.timedelta
            elif name == 'train_test_split':
                # Handle sklearn train_test_split
                success, _ = safe_auto_install_package('scikit-learn')
                if success:
                    from sklearn.model_selection import train_test_split
                    return train_test_split
                raise ImportError(f"Cannot import {name}")
            elif name == 'LinearRegression':
                # Handle sklearn LinearRegression
                success, _ = safe_auto_install_package('scikit-learn')
                if success:
                    from sklearn.linear_model import LinearRegression
                    return LinearRegression
                raise ImportError(f"Cannot import {name}")
            elif name == 'mean_squared_error':
                # Handle sklearn mean_squared_error
                success, _ = safe_auto_install_package('scikit-learn')
                if success:
                    from sklearn.metrics import mean_squared_error
                    return mean_squared_error
                raise ImportError(f"Cannot import {name}")
            elif name == 'r2_score':
                # Handle sklearn r2_score
                success, _ = safe_auto_install_package('scikit-learn')
                if success:
                    from sklearn.metrics import r2_score
                    return r2_score
                raise ImportError(f"Cannot import {name}")
            
            # ABSOLUTE WILDCARD: Try to install ANY package safely
            success, message = safe_auto_install_package(name)
            if success:
                try:
                    module = __import__(name)
                    return module
                except ImportError:
                    pass
            
            raise NameError(f"name '{name}' is not defined")
    
    # Add dynamic importer for missing names
    class SmartGlobals(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.importer = DynamicImporter()
        
        def __getitem__(self, key):
            try:
                return super().__getitem__(key)
            except KeyError:
                try:
                    value = self.importer[key]
                    self[key] = value  # Cache for future use
                    return value
                except (ImportError, NameError):
                    raise NameError(f"name '{key}' is not defined")
    
    return SmartGlobals(globals_dict)

def clean_code(code: str) -> str:
    """
    Remove Markdown triple backticks and optional 'python' language tag
    from the start and end of LLM-generated code snippets.
    Ensures valid Python syntax with proper indentation.
    """
    # Remove opening triple backticks with optional 'python' and optional newline
    code = re.sub(r"^```(?:python)?\n?", "", code, flags=re.IGNORECASE)
    # Remove trailing triple backticks, possibly preceded by a newline
    code = re.sub(r"\n?```$", "", code)
    
    # Clean up the code
    code = code.strip()
    
    # Validate syntax by trying to compile
    try:
        compile(code, '<string>', 'exec')
        return code
    except SyntaxError as e:
        # If syntax error, try to fix common issues
        pass
    
    # If we get here, there was a syntax error. Try to extract and fix the function
    lines = code.split('\n')
    
    # Find the function definition
    function_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def answer_question(') and line.strip().endswith(':'):
            function_start = i
            break
    
    if function_start == -1:
        # No valid function found, create a minimal one
        return "def answer_question(df):\n    return 'No valid function found'"
    
    # Extract the function
    function_line = lines[function_start]
    base_indent = len(function_line) - len(function_line.lstrip())
    
    # Start with the function definition
    result_lines = [function_line]
    
    # Add all properly indented lines that follow
    for i in range(function_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            result_lines.append('')
            continue
            
        # Calculate indentation
        line_indent = len(line) - len(line.lstrip())
        
        # Lines that are part of the function body should be indented more than the function definition
        if line_indent > base_indent:
            result_lines.append(line)
        else:
            # End of function
            break
    
    result = '\n'.join(result_lines)
    
    # Final validation - try to compile again
    try:
        compile(result, '<string>', 'exec')
        return result
    except SyntaxError:
        # If still broken, return a safe fallback
        return f"""def answer_question(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        # Original code had syntax issues, returning error message
        return "Code execution failed due to syntax errors"
    except Exception as e:
        return str(e)"""

def find_closest_value(target: str, choices: pd.Series, threshold: int = 70) -> str:
    """
    Find closest string in 'choices' to 'target' using fuzzy matching.
    Return best match if above threshold; else return original target.
    """
    unique_choices = choices.dropna().unique().tolist()
    if not unique_choices:
        return target
    match, score = process.extractOne(target, unique_choices)
    if score >= threshold:
        return match
    else:
        return target

def patch_code_filter_values(code: str, df: pd.DataFrame) -> str:
    """
    Scan the generated code for filter strings and replace them with best dataset values.
    Looks for patterns like df['col'] == 'value' and replaces 'value' with closest match.
    """
    pattern = re.compile(r"(df\[['\"](\w+)['\"]\]\s*==\s*['\"]([^'\"]+)['\"])")


    def replacer(match):
        full_match = match.group(1)
        col = match.group(2)
        val = match.group(3)

        if col in df.columns:
            new_val = find_closest_value(val, df[col])
            if new_val != val:
                replaced = f"df['{col}'] == '{new_val}'"
                print(f"[Info] Replacing filter value '{val}' with closest dataset value '{new_val}' in column '{col}'")
                return replaced
        return full_match


    patched_code = pattern.sub(replacer, code)
    return patched_code

class GenericLLMCodeExecutor:
    def __init__(self, api_key=None, model="gpt-4"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer_code(self, df: pd.DataFrame, question: str, max_rows=10):
        sample_rows = df.head(max_rows).to_dict(orient="records")
        columns = df.columns.tolist()

        prompt = f"""You are a Python data analyst. You have a pandas DataFrame 'df' with columns:

{columns}

Here are {max_rows} sample rows from 'df':
{json.dumps(sample_rows, indent=2)}

Write a Python function called `answer_question(df)` that:
- Uses the pandas DataFrame `df` (full data, not just sample) to answer this question:

\"\"\"{question}\"\"\"

- Returns the answer (string, number, list, dict, or DataFrame).
- For numerical results, convert numpy types to native Python types (use .item() for scalars, .tolist() for arrays).
- For visualization requests: Create the plot/chart and return a descriptive message like "Time series plot created" or "Heatmap visualization generated".
- For plots: Use matplotlib.pyplot as plt, seaborn as sns, or plotly as appropriate.
- Ensure plots have proper titles, axis labels, and formatting.
- For matplotlib/seaborn plots: Use plt.show() to display the plot.
- Do NOT include data loading or print statements.

DATA HANDLING GUIDELINES:
- If columns have numeric names, the data might be transposed or have header issues
- For stock data, look for columns that contain price-like values (usually the 4th or 5th column)
- Always add error handling with try/except blocks
- Check array sizes before indexing (e.g., if len(array) > 0 before accessing array[0])
- Use .iloc[] for positional indexing when column names are unclear

DATE HANDLING GUIDELINES:
- When working with dates, always use pandas.to_datetime() to convert strings to datetime objects
- For date arithmetic, use pandas.Timedelta or datetime.timedelta for adding/subtracting days
- When predicting future dates: last_date + pandas.Timedelta(days=100) or last_date + datetime.timedelta(days=100)
- Convert datetime to ordinal for ML models: date.toordinal() method
- Always handle date parsing errors with try/except blocks
- For stock prediction, parse the first column as dates if it contains date-like strings

MACHINE LEARNING GUIDELINES:
- For time series prediction, convert dates to ordinal numbers for model input
- Use train_test_split for proper data splitting
- Always include model evaluation metrics (R², MSE, etc.)
- Handle feature scaling if necessary with StandardScaler
- For stock prediction, consider using multiple features if available (open, high, low, volume)

IMPORT GUIDELINES:
- For FFT: Use "import numpy" then "numpy.fft.fft()" for clarity
- For scientific computing: Use "import numpy" then "numpy.array()"
- For plotting: Use "import matplotlib.pyplot" then "matplotlib.pyplot.plot()"
- For statistics: Use "import scipy.stats" then "scipy.stats.norm()"
- For date handling: Use "import pandas" then "pandas.to_datetime()" and "pandas.Timedelta()"
- For machine learning: Use "from sklearn.model_selection import train_test_split"
- Avoid from-imports when possible (e.g., avoid "from numpy import fft")
- Use explicit imports to prevent naming conflicts

FFT PLOTTING GUIDELINES:
- For FFT plots, x-axis should be FREQUENCIES (use numpy.fft.fftfreq() to get frequency bins)
- For FFT plots, y-axis should be AMPLITUDES (absolute values of FFT coefficients)
- Do NOT plot FFT amplitudes against time/dates - this is incorrect
- Use fftfreq to get proper frequency values for x-axis
- Consider using only positive frequencies (first half of FFT result) for cleaner visualization

CRITICAL: Respond with ONLY the Python function code. No explanations, no comments outside the function, no text before or after the code. Start with 'def answer_question(df):' and end with the return statement. Use proper 4-space indentation for all code inside the function.
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You write Python functions for data analysis and visualization. Always create well-formatted plots with proper titles and labels."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1200,
        )
        code = completion.choices[0].message.content.strip()
        return code

    def execute_code(self, code: str, df: pd.DataFrame):
        local_env = {}
        try:
            clean = clean_code(code)
            patched = patch_code_filter_values(clean, df)
            # Disable import patching - rely on dynamic globals instead
            # The dynamic globals system is comprehensive enough to handle package imports
            
            # Use dynamic globals that auto-install packages
            smart_globals = create_dynamic_globals()
            
            # Add essential imports to globals
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            import matplotlib.pyplot as plt
            smart_globals['pd'] = pd
            smart_globals['plt'] = plt
            
            exec(patched, smart_globals, local_env)
            if "answer_question" not in local_env:
                return None, "Generated code does not define 'answer_question(df)'."

            result = local_env["answer_question"](df)
            
            # Convert numpy types to native Python types for cleaner display
            result = convert_numpy_types(result)
            
            # If result is None but there are matplotlib figures, return a success message
            if result is None and plt.get_fignums():
                result = "Visualization created successfully"
            
            return result, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"Error while executing code: {e}\n{tb}"

    def generate_explanation(self, question: str, answer, code: str, df: pd.DataFrame = None):
        dataset_overview = ""
        if df is not None:
            dataset_overview = f"\n\nDataset columns: {', '.join(df.columns)}\nSample rows:\n{df.head(3).to_markdown(index=False)}\n"
        prompt = f"""You are a data analyst.

Question asked:
{question}

Answer computed:
{json.dumps(answer, indent=2)}

{dataset_overview}
Please provide a detailed explanation of the analysis performed to reach this conclusion.
Include:
- The specific data columns or features used from the dataset,
- How these columns were analyzed or combined in the code,
- How the code logic relates to the dataset structure and the research question,
- Any normalization or scoring approach,
- Why the chosen result is meaningful,
- Any assumptions or limitations.

Here is the Python code used for the analysis:

{code}

Explain the code's logic, how it interacts with the dataset, and how it supports the answer.
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You explain data analysis results clearly and thoroughly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=700,
        )
        explanation = completion.choices[0].message.content.strip()
        return explanation

def main():
    csv_path = input("Enter path to your dataset (CSV): ").strip()
    question = input("Enter your question about the dataset: ").strip()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    executor = GenericLLMCodeExecutor()

    print("\nGenerating Python code to answer your question...")
    code = executor.generate_answer_code(df, question)
    print("\n=== Generated Python function ===\n")
    print(code)

    print("\nExecuting generated code on your dataset...")
    answer, error = executor.execute_code(code, df)

    if error:
        print(f"\nError during code execution:\n{error}")
        return

    print("\n=== Raw Answer from Executed Code ===\n")
    print(answer)

    print("\nGenerating detailed explanation for the analysis...\n")
    explanation = executor.generate_explanation(question, answer, code)
    print(explanation)

if __name__ == "__main__":
    main()
