
import re
import json

def parse_generated_response(example):
    response = example.get("response", "")
    if not response:
        return None

    # Regex to match content between ```json and ```
    match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if not match:
        return None

    json_str = match.group(1).strip()
    if not json_str:
        print("Empty JSON content in response")
        return None

    try:
        data = json.loads(json_str)
        return {"parsed": data}
    except json.JSONDecodeError:
        print("Invalid JSON content in response")
        return None


def parse_analysis_content(example):
    parsed_data = example.get("parsed")
    if parsed_data is None:
        return None

    description = parsed_data.get("description", "")
    explanation = parsed_data.get("explanation", [])
    final_answer = parsed_data.get("final_answer", "")

    explanation_steps = []
    if isinstance(explanation, list):
        explanation_steps = [step.strip() for step in explanation if isinstance(step, str)]
    elif isinstance(explanation, str):
        explanation_steps = [line.strip("- ").strip() for line in explanation.split("\n") if line.strip()]

    return {
        "description": description,
        "explanation_steps": explanation_steps,
        "final_answer": final_answer
    }