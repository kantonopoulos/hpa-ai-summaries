import json
import requests

def baseline_gene_summary(gene_json):
    """
    Calls a local LLM API to generate a summary for a gene given its JSON data.
    """
    prompt = f"Write a concise, single-paragraph summary for this gene based on the following JSON data: {json.dumps(gene_json)}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gpt-oss:20b", "prompt": prompt},
        stream=True
    )
    result = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode())
            if "response" in data:
                result += data["response"]
    return result