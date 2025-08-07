import os
import re
import json
import traceback
import subprocess
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from thefuzz import process

load_dotenv()

def auto_install_package(package_name):
    """
    Automatically install a package if it's not available.
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        try:
            print(f"Installing missing package: {package_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install package: {package_name}")
            return False

def patch_imports_in_code(code):
    """
    Patch import statements in code to handle package installation.
    """
    lines = code.split('\n')
    patched_lines = []
    
    package_mappings = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'skimage': 'scikit-image'
    }
    
    for line in lines:
        original_line = line
        stripped = line.strip()
        
        # Handle "import sklearn" or "from sklearn import ..."
        if stripped.startswith('import sklearn') or stripped.startswith('from sklearn'):
            patched_lines.append(f"""
try:
    {original_line}
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    {original_line}""")
        # Handle other common packages
        elif any(f'import {pkg}' in stripped or f'from {pkg}' in stripped for pkg in package_mappings.keys()):
            for pkg, real_name in package_mappings.items():
                if f'import {pkg}' in stripped or f'from {pkg}' in stripped:
                    patched_lines.append(f"""
try:
    {original_line}
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "{real_name}"])
    {original_line}""")
                    break
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
        if auto_install_package(package_name):
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
            if auto_install_package(package_name):
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
    }
    
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
    
    class DynamicImporter:
        def __getitem__(self, name):
            # Handle special statistical functions
            if name == 'adfuller':
                if auto_install_package('statsmodels'):
                    from statsmodels.tsa.stattools import adfuller
                    return adfuller
                raise ImportError(f"Cannot import {name}")
            elif name == 'kpss':
                if auto_install_package('statsmodels'):
                    from statsmodels.tsa.stattools import kpss
                    return kpss
                raise ImportError(f"Cannot import {name}")
            elif name == 'seasonal_decompose':
                if auto_install_package('statsmodels'):
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    return seasonal_decompose
                raise ImportError(f"Cannot import {name}")
            elif name == 'stats':
                if auto_install_package('scipy'):
                    import scipy.stats as stats
                    return stats
                raise ImportError(f"Cannot import {name}")
            elif name == 'sklearn':
                if auto_install_package('scikit-learn'):
                    import sklearn
                    return sklearn
                raise ImportError(f"Cannot import {name}")
            
            # Try direct import first
            module = ensure_package_available(name)
            if module:
                return module
            
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
    """
    # Remove opening triple backticks with optional 'python' and optional newline
    code = re.sub(r"^```(?:python)?\n?", "", code, flags=re.IGNORECASE)
    # Remove trailing triple backticks, possibly preceded by a newline
    code = re.sub(r"\n?```$", "", code)
    return code.strip()

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

- Returns the answer (string, number, list, or dict).
- Do NOT include data loading or print statements.
- ONLY provide the function code without any explanations or text.
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You write Python functions for data analysis."},
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
            # Also patch import statements
            import_patched = patch_imports_in_code(patched)
            
            # Use dynamic globals that auto-install packages
            smart_globals = create_dynamic_globals()
            
            exec(import_patched, smart_globals, local_env)
            if "answer_question" not in local_env:
                return None, "Generated code does not define 'answer_question(df)'."

            result = local_env["answer_question"](df)
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
