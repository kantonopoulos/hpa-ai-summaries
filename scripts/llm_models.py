import json
import requests

def baseline_gene_summary(gene_json):
    """
    Calls a local LLM API to generate a summary for a gene given its JSON data.
    """
    prompt = f"Summarize and comment the underlining trends (30-50 words): {json.dumps(gene_json)}"
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

def concise_gene_summary(gene_json):
    """
    Generates a concise summary (30-50 words, max 100) focusing on:
    Tissue/Cell type specificity, Subcellular location, diseases, pathways, and blood information.
    """
    prompt = (
        "Act as an expert biologist and scientific writer. "
        "Write a concise summary (30-50 words, max 100) for this gene. "
        "Focus on Tissue/Cell type specificity, Subcellular location, diseases, pathways, and blood information (donot mention methods). "
        "Try to use the important information as it is from the data (gene name, tissue/cell types, subcellular location, diseases), if they exist. "
        f"Here is the gene data: {json.dumps(gene_json)}"
    )
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

def refined_gene_summary(concise_summary, gene_json):
    """
    Refines the concise summary by integrating key results with concise comments
    to highlight underlying trends and connections. Produces a single cohesive paragraph
    that combines the raw summary with insights, avoiding reiteration or a "summary + comments" structure.
    """
    prompt = (
        "You are an expert molecular biologist and Human Protein Atlas researcher. "
        "Integrate the key results from the summary without changing them with concise, insightful comments that highlight underlying trends, "
        "connections between Tissue/Cell type specificity, Subcellular location, diseases, and pathways, "
        "and any relevant scientific literature. Avoid creating a 'summary + comments' structure. "
        "Focus on producing a clear, concise, and scientifically accurate paragraph that integrates the information seamlessly. "
        "Use direct, easy to understand language suitable for a scientific audience. "
        "Do not add unsupported claims or unnecessary elaboration. Keep the refined summary between 30-50 words (strictly under 100). "
        f"Concise summary: {concise_summary}\nGene data: {json.dumps(gene_json)}"
    )
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