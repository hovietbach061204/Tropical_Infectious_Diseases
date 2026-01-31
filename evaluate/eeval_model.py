import os
import re
import sys
import ollama
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from Generate_Response.trindsLangchain import generate_response

load_dotenv()

def extract_diagnosis(text: str) -> str:
    """Extracts the diagnosis from the model's output string."""
    m = re.search(r"Diagnosis:\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return ""
    diag = m.group(1).strip()
    diag = diag.split("Explanation:")[0].strip()
    return diag

def get_model_answer(symptoms: str) -> str:
    """Queries the local Ollama model."""
    system_prompt = """
You are an expert in tropical and infectious diseases.

Return your answer in this format (exactly two lines):
Diagnosis: <disease name>
Explanation: <brief explanation>
""".strip()

    resp = ollama.chat(
        model=os.getenv("CHAT_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": symptoms},
        ],
    )
    return resp["message"]["content"]

def get_model_answer_v2(question: str) -> str:
    response = generate_response(question)
    return response

def main():
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. DeepEval will call OpenAI for judging.")
    
    while True:
        print("\n\n-------------------------------")
        question = input(">>> ")
        print("\n\n")
        if question == "/bye":
            break
        expected_ground_truth = input("Expected Diagnosis: ")

        print(f"Generating answer for: {expected_ground_truth}...")
    
        # 1. Get Actual Output from Ollama
        actual_output = get_model_answer(question)
        # --- PRINT TO TERMINAL (Added this) ---
        print("=" * 10)
        print("MODEL GENERATED OUTPUT:")
        print("-" * 10)
        print(actual_output)
        print("=" * 10)
        print("\nRunning DeepEval Judges...\n")
        # --------------------------------------
        
        # 2. Extract specific diagnosis to compare
        predicted_diag = extract_diagnosis(actual_output) or "UNKNOWN"
        diagnosis_only_actual = f"Diagnosis: {predicted_diag}"
        diagnosis_only_expected = f"Diagnosis: {expected_ground_truth}"

        # 3. Create the Single Test Case
        test_case = LLMTestCase(
            input=question,
            actual_output=diagnosis_only_actual,
            expected_output=diagnosis_only_expected
        )

        # ---------------------------------------------------------
        # METRICS DEFINITION
        # ---------------------------------------------------------
        
        # Metric 1: Diagnosis correctness
        correctness = GEval(
            name="Diagnosis Correctness",
            model=os.getenv("OPENAI_JUDGE_MODEL"),
            criteria=(
                "Look for the line starting with 'Diagnosis:'. "
                "Score whether the ACTUAL diagnosis refers to the same underlying condition as the EXPECTED diagnosis. "
                "Do NOT require an exact string match. "
                "If the diagnosis is a synonym, a medical variation, or semantically equivalent (e.g., 'Enteric Fever' vs 'Typhoid'), score it as 1.0. "
                "Ignore the explanation section for this score. "
                "0.0 = completely unrelated or incorrect disease."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            strict_mode=True,
        )

        # Metric 2: Reasoning quality
        reasoning = GEval(
            name="Reasoning Quality",
            model=os.getenv("OPENAI_JUDGE_MODEL"),
            criteria=(
                "Score the quality of the 'Explanation' section. "
                "It must reference patient symptoms/exposures (cycles of fever, travel history), "
                "be medically logical based on the retrieved context, and justify the diagnosis."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            strict_mode=False,
        )

        # ---------------------------------------------------------
        # RUN EVALUATION
        # ---------------------------------------------------------
        # Note: test_cases expects a list, so we put our single case in []
        evaluate(test_cases=[test_case], metrics=[correctness, reasoning])
    

if __name__ == "__main__":
    main()