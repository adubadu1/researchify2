import os
import re
import json
import traceback
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from thefuzz import process

load_dotenv()

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
            exec(patched, {"pd": pd, "np": np}, local_env)
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
