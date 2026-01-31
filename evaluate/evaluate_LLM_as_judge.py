# evaluate_with_perplexity_judge.py
# Generator: local Ollama model
# Judge: Perplexity Sonar via OpenAI-compatible API
# Metrics: diagnostic accuracy, explanation quality, hallucination, clinical relevance, reasoning quality

import csv
import re
import os
import json
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv
import ollama
from openai import OpenAI
from Generate_Response.trindsLangchain import generate_response

load_dotenv()

CSV_PATH = "/Volumes/SandiskSSD/Downloads/Tropical-Dataset-ED.csv"
SAMPLE_SIZE = 50  # set 50–100 later


# Read key from environment (recommended)
# PowerShell:  $env:PERPLEXITY_API_KEY="pplx-..."
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    raise RuntimeError("Missing PERPLEXITY_API_KEY in environment.")

PPLX_MODEL = "sonar"  # permitted Sonar model name for chat completions [web:165]

@dataclass
class TestCase:
    case_id: int
    question: str
    ground_truth: str
    actual_output: str
    predicted_diag: str


def extract_diagnosis(text: str) -> str:
    match = re.search(r"Diagnosis:\s*(.+)", text, re.IGNORECASE)
    if not match:
        return ""
    line = match.group(1).strip()
    line = line.split("Explanation:")[0].strip()
    return line

# def extract_explanation(text: str) -> str:


def get_model_answer(symptoms: str) -> str:
    system_prompt = """
You are an expert in tropical and infectious diseases.

Return your answer in this format (exactly two lines):
Diagnosis: <disease name>
Explanation: <brief explanation>
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": symptoms},
    ]

    response = ollama.chat(
        model="unsloth_llama-3.1-medical:latest",
        messages=messages,
    )
    return response["message"]["content"]


def get_model_answer_v2(question: str) -> str:
    response = generate_response(question)
    return response


def parse_score_json(raw: str) -> Dict[str, Any]:
    """
    Robust parsing for judge outputs:
    - strips ```json ... ``` fences
    - extracts the first {...} block
    - loads JSON
    """
    if raw is None:
        raise ValueError("Empty judge output")

    text = raw.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # Extract the first JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in: {text[:200]!r}")

    json_str = text[start : end + 1].strip()

    # Fix trailing commas (common LLM issue)
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    obj = json.loads(json_str)

    # Normalize expected fields
    if "score" not in obj:
        raise ValueError(f"Missing 'score' in JSON: {obj}")
    if "reason" not in obj:
        obj["reason"] = ""

    # Ensure score is float
    obj["score"] = float(obj["score"])
    obj["reason"] = str(obj["reason"])
    return obj


class PerplexityJudge:
    def __init__(self):
        self.client = OpenAI(
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        )

    def _judge(
        self,
        metric_name: str,
        question: str,
        expected: str,
        actual: str,
        system_prompt: str,
        example_json: str,
        debug: bool = True,
    ) -> Dict[str, Any]:

        expected_block = f"\nExpected diagnosis:\n'{expected}'\n" if expected else ""

        user_prompt = f"""
Patient:
"{question}"
{expected_block}
Model answer:
"{actual}"

Return ONLY valid JSON like:
{example_json}
""".strip()

        try:
            resp = self.client.chat.completions.create(
                model=PPLX_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=220,
            )

            content = resp.choices[0].message.content.strip()
            if debug:
                print(f"DEBUG [{metric_name}] raw response: {repr(content[:180])}...")

            parsed = parse_score_json(content)
            return parsed

        except Exception as e:
            if debug:
                print(f"DEBUG [{metric_name}] judge error: {type(e).__name__}: {e}")
            return {"score": 0.0, "reason": f"Judge error: {type(e).__name__}: {e}"}

    def judge_diagnostic_accuracy(self, question: str, expected: str, actual: str) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation model. Output ONLY JSON: {"score": float 0.0-1.0, "reason": string}.

Score criteria:
- 1.0: Diagnosis matches expected diagnosis (minor spelling OK).
- 0.5: Related disease family but not the expected disease.
- 0.0: Incorrect diagnosis.
"""
        return self._judge(
            metric_name="diag",
            question=question,
            expected=expected,
            actual=actual,
            system_prompt=system_prompt,
            example_json='{"score": 0.8, "reason": "short explanation"}',
        )

    def judge_explanation_quality(self, question: str, actual: str) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation model. Output ONLY JSON: {"score": float 0.0-1.0, "reason": string}.
Evaluate on the 'Explanation' in the model's answer with the patient question.
Score criteria:
- 1.0: Clear explanation, references key symptoms/exposures.
- 0.0: Not addressing symptoms or unclear. 
"""
        return self._judge(
            metric_name="expl",
            question=question,
            expected="",
            actual=actual,
            system_prompt=system_prompt,
            example_json='{"score": 0.9, "reason": "short explanation"}',
        )

    def judge_hallucination(self, question: str, actual: str) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation model. Output ONLY JSON: {"score": float 0.0-1.0, "reason": string}.

1.0 = no unsupported facts relative to the patient description.
0.0 = many unsupported facts.
"""
        return self._judge(
            metric_name="hall",
            question=question,
            expected="",
            actual=actual,
            system_prompt=system_prompt,
            example_json='{"score": 1.0, "reason": "short explanation"}',
        )


    def judge_clinical_relevance(self, question: str, actual: str) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation model. Output ONLY JSON: {"score": float 0.0-1.0, "reason": string}.

1.0 = diagnosis fits symptoms/exposure/location.
0.0 = does not fit clinically/epidemiologically.
"""
        return self._judge(
            metric_name="clin",
            question=question,
            expected="",
            actual=actual,
            system_prompt=system_prompt,
            example_json='{"score": 0.95, "reason": "short explanation"}',
        )

    def judge_reasoning_quality(self, question: str, actual: str) -> Dict[str, Any]:
        system_prompt = """
You are an evaluation model. Output ONLY JSON: {"score": float 0.0-1.0, "reason": string}.
Evaluate on the 'Explanation' in the model's answer with the patient question.
1.0 = logical medical reasoning.
0.0 = illogical reasoning.
"""
        return self._judge(
            metric_name="reas",
            question=question,
            expected="",
            actual=actual,
            system_prompt=system_prompt,
            example_json='{"score": 0.85, "reason": "short explanation"}',
        )


def main():
    print(f"Evaluating {SAMPLE_SIZE} cases using Perplexity judge (model={PPLX_MODEL})...")

    judge = PerplexityJudge()
    results = []

    with open(CSV_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= SAMPLE_SIZE:
                break

            question_text = row["question_text"].strip()
            ground_truth = row["ground truth"].strip()
            if not question_text or not ground_truth:
                continue

            print(f"\nCase {i+1}/{SAMPLE_SIZE}: {question_text[:80]}...")

            actual_output = get_model_answer(question_text) # Diagnosis + Explanation
            predicted_diag = extract_diagnosis(actual_output) # Diagnosis only

            diag_only = f"Diagnosis: {predicted_diag}"


            diag = judge.judge_diagnostic_accuracy(question_text, ground_truth, diag_only)
            clin = judge.judge_clinical_relevance(question_text, diag_only)
            expl = judge.judge_explanation_quality(question_text, actual_output)
            hall = judge.judge_hallucination(question_text, actual_output)
            reas = judge.judge_reasoning_quality(question_text, actual_output)

            print(f"Predicted: {predicted_diag}")
            print(f"Diag: {diag['score']:.2f} | {diag['reason'][:80]}...")
            print(f"Expl: {expl['score']:.2f} | {expl['reason'][:80]}...")
            print(f"Hall: {hall['score']:.2f} | {hall['reason'][:80]}...")
            print(f"Clin: {clin['score']:.2f} | {clin['reason'][:80]}...")
            print(f"Reas: {reas['score']:.2f} | {reas['reason'][:80]}...")

            results.append({
                "case": i + 1,
                "question": question_text,
                "expected": ground_truth,
                "predicted": predicted_diag,
                "diag_score": diag["score"],
                "diag_reason": diag["reason"],
                "expl_score": expl["score"],
                "expl_reason": expl["reason"],
                "hall_score": hall["score"],
                "hall_reason": hall["reason"],
                "clin_score": clin["score"],
                "clin_reason": clin["reason"],
                "reas_score": reas["score"],
                "reas_reason": reas["reason"],
            })

    if not results:
        print("No cases evaluated.")
        return

    avg = lambda k: sum(r[k] for r in results) / len(results)
    print(f"Model's output:\n{actual_output}\n")
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total cases: {len(results)}")
    print(f"Diagnostic accuracy:  {avg('diag_score'):.3f}")
    print(f"Explanation quality:  {avg('expl_score'):.3f}")
    print(f"Hallucination:        {avg('hall_score'):.3f}")
    print(f"Clinical relevance:   {avg('clin_score'):.3f}")
    print(f"Reasoning quality:    {avg('reas_score'):.3f}")

    out_csv = "./fine_tuned_model/unsloth_llama_31.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()