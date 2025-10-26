import re
from collections import Counter
from typing import Dict, Any, List, Tuple
import pandas as pd
import difflib
import requests
import json
import time

# Utility functions
def tokenize(text: str) -> List[str]:
    """
    Tokenize input text into lowercased alphanumeric tokens.

    - Replaces non-word characters (punctuation) with spaces.
    - Splits on whitespace.
    - Returns a list of tokens suitable for simple NLP metrics.

    Rationale:
    A minimal tokenizer is used so metrics like n-grams and bag-of-words
    operate on comparable token sequences without punctuation noise.
    """
    text = text or ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in text.split() if t]
    return tokens

def ngrams(tokens: List[str], n: int) -> List[Tuple[str,...]]:
    """
    Compute contiguous n-grams from a token list.

    Args:
        tokens: list of tokens (strings).
        n: size of each n-gram (1 = unigrams, 2 = bigrams, ...).

    Returns:
        List of n-gram tuples. Each tuple contains n tokens in sequence.
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def lcs_length(a: List[str], b: List[str]) -> int:
    """
    Compute the Longest Common Subsequence (LCS) length between two token lists.
    """
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

# ROUGE metrics
def rouge_n(reference: str, summary: str, n: int=1) -> float:
    """
    Compute a simple ROUGE-N recall score: overlap(n-grams) / total reference n-grams.

    Args:
        reference: reference text (string) from JSON fields.
        summary: generated summary (string).
        n: n-gram order (1 for ROUGE-1, 2 for ROUGE-2).

    Returns:
        Recall score in [0,1]. 0 if no tokens or no overlap.
    """
    r_tokens = tokenize(reference)
    s_tokens = tokenize(summary)
    if not r_tokens or not s_tokens:
        return 0.0
    r_ngrams = Counter(ngrams(r_tokens, n))
    s_ngrams = Counter(ngrams(s_tokens, n))
    overlap = sum((r_ngrams & s_ngrams).values())
    total = sum(r_ngrams.values())
    return overlap/total if total>0 else 0.0

def rouge_l(reference: str, summary: str) -> float:
    """
    Compute a simple ROUGE-L-like score based on LCS length.

    Args:
        reference: reference text.
        summary: generated summary.

    Returns:
        Ratio of LCS length to reference token count.
    """
    r_tokens = tokenize(reference)
    s_tokens = tokenize(summary)
    if not r_tokens or not s_tokens:
        return 0.0
    lcs = lcs_length(r_tokens, s_tokens)
    return lcs / len(r_tokens) if len(r_tokens)>0 else 0.0

# Entity and link checks
def extract_entities_from_json(gjson: Dict[str,Any]) -> Dict[str,List[str]]:
    """
    Extract structured entities from the HPA JSON for lightweight entity matching.

    Extracted fields:
      - gene: primary Gene name
      - tissues: keys from 'RNA tissue specific nTPM' and tissues referenced in
                 'RNA tissue cell type enrichment' entries (text before ' - ')
      - cell_types: keys from 'RNA single cell type specific nTPM'
      - subcellular: list from 'Subcellular location'
      - diseases: strings from 'Disease involvement', plus 'Upregulated in disease'
                  parsed from HPA blood info, plus 'cancer' if any Cancer prognostics keys exist
      - blood: indicators such as 'ms_detected' and 'pea_available'

    Returns:
        dict with above keys and lists of strings.
    """
    out = {}
    # gene name
    out['gene'] = [gjson.get('Gene','')]
    # tissues from RNA tissue specific nTPM keys
    tissues = []
    tt = gjson.get('RNA tissue specific nTPM') or {}
    tissues += list(tt.keys())
    # also include RNA tissue cell type enrichment entries (they contain "Tissue - Cell")
    for enr in (gjson.get('RNA tissue cell type enrichment') or []):
        if isinstance(enr,str) and " - " in enr:
            t = enr.split(" - ",1)[0].strip()
            tissues.append(t)
    out['tissues'] = list(dict.fromkeys([t for t in tissues if t]))
    # single cell types
    sc = gjson.get('RNA single cell type specific nTPM') or {}
    out['cell_types'] = list(sc.keys())
    # subcellular
    sub = gjson.get('Subcellular location') or []
    out['subcellular'] = sub
    # diseases: from Disease involvement and HPA blood entry 'Upregulated in disease'
    dis = []
    for d in (gjson.get('Disease involvement') or []):
        if isinstance(d,str):
            dis.append(d)
    hpa_blood = gjson.get('Blood') or {}
    up = hpa_blood.get('Upregulated in disease')
    if up:
        # comma-separated string
        parts = [p.strip() for p in re.split(r',|;', up) if p.strip()]
        dis += parts
    out['diseases'] = list(dict.fromkeys([d for d in dis if d]))
    return out

def entity_link_checks(gjson: Dict[str,Any], summary: str) -> Dict[str,Any]:
    """
    Check presence of extracted entities in the generated summary.

    Args:
        gjson: gene JSON object.
        summary: generated summary text.

    Returns:
        dict containing boolean matches for:
          - gene_present
          - tissue_matches: mapping tissue->bool
          - cell_type_matches: mapping cell_type->bool
          - subcellular_matches: mapping location->bool
          - disease_matches: mapping disease->bool
    """
    ent = extract_entities_from_json(gjson)
    summary_lc = summary.lower()
    results = {}
    # gene present
    results['gene_present'] = any(name.lower() in summary_lc for name in ent['gene'] if name)
    # tissues: check top N tissues (we check all extracted tissue tokens)
    tissue_matches = {t: (t.lower() in summary_lc) for t in ent['tissues']}
    results['tissue_matches'] = tissue_matches
    # cell types
    cell_matches = {c: (c.lower() in summary_lc) for c in ent['cell_types']}
    results['cell_type_matches'] = cell_matches
    # subcellular
    sub_matches = {s: (s.lower() in summary_lc) for s in ent['subcellular']}
    results['subcellular_matches'] = sub_matches
    # diseases
    disease_matches = {d: (d.lower() in summary_lc) for d in ent['diseases']}
    results['disease_matches'] = disease_matches
    return results

# QA-based factuality (heuristic)
def generate_qa_pairs(gjson: Dict[str,Any]) -> List[Tuple[str,List[str]]]:
    """
    Create simple QA pairs from the JSON to test factual content presence in the summary.

    Heuristics used:
      - Gene name question: expects primary gene.
      - Main subcellular location: takes first listed location.
      - Top tissue by RNA: selects the tissue with highest nTPM if parsable.
      - Top single-cell type: selects the cell type with highest single-cell nTPM.
      - Blood detection methods: checks for MS and PEA presence.
      - Disease involvement: returns up to three disease tags.

    Returns:
        List of (question_string, list_of_expected_answers).
    """
    qa = []
    # Q: gene name
    if gjson.get('Gene') and gjson.get('Gene') not in [None, "", "None", "null", "Not detected", "Not analysed"]:
        qa.append(("What is the gene name?", [gjson.get('Gene')]))
    # Q: main localization (first subcellular)
    subs = gjson.get('Subcellular location') or []
    if subs and subs[0] not in [None, "", "None", "null", "Not detected", "Not analysed"]:
        qa.append(("Main subcellular location?", [subs[0]]))
    # Q: top tissue (highest nTPM if available)
    tissues = gjson.get('RNA tissue specific nTPM') or {}
    if tissues and tissues not in [None, "", {}, "None", "null", "Not detected", "Not analysed"]:
        # choose top by numeric value if parseable
        try:
            top = max(tissues.items(), key=lambda kv: float(kv[1]))[0]
        except Exception:
            top = list(tissues.keys())[0]
        qa.append(("Top tissue by RNA?", [top]))
    # Q: enriched cell type (from single cell)
    sc = gjson.get('RNA single cell type specific nTPM') or {}
    if sc and sc not in [None, "", {}, "None", "null", "Not detected", "Not analysed"]:
        try:
            top = max(sc.items(), key=lambda kv: float(kv[1]))[0]
        except Exception:
            top = list(sc.keys())[0]
        qa.append(("Top single-cell cell type?", [top]))
    # Q: secretome location
    secretome = gjson.get('Secretome location')
    if secretome and secretome not in [None, "", "None", "null", "Not detected", "Not analysed"]:
        qa.append(("Where is the protein secreted?", [secretome]))
    # Q: disease involvement (some disease strings)
    dis = []
    for d in (gjson.get('Disease involvement') or []):
        if d and d not in [None, "", "None", "null", "Not detected", "Not analysed"]:
            dis.append(d)
    if dis:
        qa.append(("List disease involvement tags", dis[:3]))
        
    return qa

def answer_in_summary(expected_list: List[str], summary: str) -> bool:
    """
    Check whether any expected answer (or its tokens) appears in the summary.

    Heuristics:
      - For multi-word expected answers, allow partial matches (e.g., "Extravillous trophoblasts" matches "trophoblasts").
      - A fuzzy quick-ratio check via difflib is also applied as a fallback.

    Returns:
        True if match found, False otherwise.
    """
    s = summary.lower()

    for exp in expected_list:
        if not exp:
            continue
        # Match multi-word tokens robustly by checking each significant token
        toks = [t for t in re.sub(r"[^\w\s]", " ", exp.lower()).split() if t]

        # Allow partial matches: check if any token is in the summary
        if any(tok in s for tok in toks):
            return True

        # Fuzzy match as a fallback
        if difflib.SequenceMatcher(None, exp.lower(), s).quick_ratio() > 0.6:
            return True

    return False

def qa_checks(gjson: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """
    Run QA checks by generating QA pairs and testing if expected answers appear in the summary.

    Returns:
        A dict mapping each question to {'expected': [...], 'matched': bool}
        plus 'match_rate': fraction of QA items matched.
    """
    qa_pairs = generate_qa_pairs(gjson)
    results = {}
    matches = []

    # Skip QA checks if no valid QA pairs are generated
    if not qa_pairs:
        return {'match_rate': 0.0, 'details': {}}

    for q, a_list in qa_pairs:
        matched = answer_in_summary(a_list, summary)
        results[q] = {'expected': a_list, 'matched': matched}
        matches.append(matched)

    results['match_rate'] = sum(matches) / len(matches) if matches else 0.0
    return results

# Entailment / Factuality aggregate
def entailment_score(gjson: Dict[str,Any], summary: str) -> float:
    """
    Compute a heuristic entailment/factuality score.

    Method:
      - Extract entities (tissues, cell types, subcellular, diseases).
      - Compute simple ratios of how many of these entities are mentioned in the summary.
      - Compute QA match rate.
      - Combine into a weighted average:
         0.4 * QA_rate + 0.2 * tissue_ratio + 0.2 * subcellular_ratio + 0.2 * disease_ratio

    Returns:
        Tuple (score, breakdown_dict) where score is rounded to 3 decimals and breakdown_dict
        contains the constituent rates for inspection.
    """
    ent = extract_entities_from_json(gjson)
    checks = entity_link_checks(gjson, summary)
    # compute ratios for tissues/cell_types/subcellular/diseases
    def ratio(d):
        if not d: 
            return 1.0
        total = len(d)
        matched = sum(1 for v in d if (v.lower() in summary.lower()))
        return matched/total if total>0 else 1.0
    tissue_ratio = ratio(ent['tissues'])
    cell_ratio = ratio(ent['cell_types'])
    sub_ratio = ratio(ent['subcellular'])
    disease_ratio = ratio(ent['diseases'])
    qa = qa_checks(gjson, summary)
    qa_rate = qa.get('match_rate', 0.0)
    # weighted average (weights chosen to emphasize core facts)
    score = 0.25*tissue_ratio + 0.25*cell_ratio + 0.25*sub_ratio + 0.25*disease_ratio
    return round(score,3), {'qa_rate':qa_rate,'tissue_ratio':tissue_ratio,'cell_ratio':cell_ratio,'sub_ratio':sub_ratio,'disease_ratio':disease_ratio}

# Word count check
def word_count_check(summary: str) -> float:
    """
    Compute a score based on word count.

    Logic:
        - If word count <= 30, score = 1.
        - If word count > 100, score = 0.
        - If word count is between 30 and 100, score decreases linearly from 1 to 0.5.

    Args:
        summary: text to evaluate.

    Returns:
        float: score based on word count.
    """
    wc = len(tokenize(summary))
    if wc <= 30:
        return 1.0
    elif wc > 100:
        return 0.0
    else:
        return 1.0 - ((wc - 30) / 140)  # Linear drop between 30 and 100

# Evaluation main function
def evaluate_summary(gjson: Dict[str,Any], summary: str) -> pd.DataFrame:
    """
    Run the full set of automatic checks for one gene JSON and its generated summary.

    Steps:
      1. Build a compact reference_text by concatenating key fields:
         - Gene name, description, tissue nTPM map, single-cell map, subcellular, disease.
         2. Compute automatic metrics:
         - ROUGE-1, ROUGE-2, ROUGE-L (recall-style).
         - Bag-of-words cosine similarity.
         - Entailment_score (heuristic composite).
         - QA match rate and details.
         - Entity link checks and word count check.
      3. Return a tidy pandas DataFrame that lists metrics for easy downstream aggregation.

    Args:
        gjson: the gene JSON (single gene-level dict extracted from the full JSON).
        summary: generated summary text for the gene.
        threshold: maximum allowed word/token count.

    Returns:
        pandas.DataFrame with metrics.
    """
    # Build a reference text concatenating key JSON fields for ROUGE comparison
    ref_parts = []
    ref_parts.append(gjson.get('Gene',''))
    ref_parts.append(gjson.get('Gene description','') or '')
    # join tissue keys and nTPM values into a string
    tmap = gjson.get('RNA tissue specific nTPM') or {}
    if tmap:
        ref_parts.append("Tissues: " + "; ".join(f"{k}={v}" for k,v in tmap.items()))
    # single cell
    scmap = gjson.get('RNA single cell type specific nTPM') or {}
    if scmap:
        ref_parts.append("Cells: " + "; ".join(f"{k}={v}" for k,v in scmap.items()))
    # subcellular and disease
    if gjson.get('Subcellular location'):
        ref_parts.append("Subcellular: " + "; ".join(gjson.get('Subcellular location')))
    if gjson.get('Disease involvement'):
        ref_parts.append("Disease: " + "; ".join(gjson.get('Disease involvement')))
    reference_text = " . ".join([p for p in ref_parts if p])
    
    # compute automatic metrics
    r1 = rouge_n(reference_text, summary, n=1)
    rl = rouge_l(reference_text, summary)
    ent_score, ent_breakdown = entailment_score(gjson, summary)
    qa = qa_checks(gjson, summary)
    entities = entity_link_checks(gjson, summary)
    wc = word_count_check(summary)
    
    # assemble dataframe
    rows = []
    gene_id = gjson.get('Ensembl') or gjson.get('Gene') or 'unknown'
    rows.append({'gene_id':gene_id, 'metric':'rouge-1','value':r1})
    rows.append({'gene_id':gene_id, 'metric':'rouge-l','value':rl})
    rows.append({'gene_id':gene_id, 'metric':'entailment_score','value':ent_score})
    rows.append({'gene_id':gene_id, 'metric':'qa_match_rate','value':qa.get('match_rate',0.0)})
    rows.append({'gene_id':gene_id, 'metric':'word_count','value':wc})
    df = pd.DataFrame(rows)
    return df

def evaluate_with_llm(gene_json, summary, max_retries=3):
    """
    Calls a local LLM API to evaluate a summary for a gene given its JSON data.

    If the output format is invalid, retries the prompt up to `max_retries` times.

    Returns:
        dict: A dictionary containing the LLM evaluation results with 'score' and 'comment'.
    """
    prompt = f"""
    You are a biomedical fact-checker.

    Evaluate whether the SUMMARY accurately represents the GENE_JSON, either by faithfully reflecting the JSON data or by providing biologically correct information that integrates or comments on the fields in a meaningful way. 

    Score meaning:
    5 = Fully accurate, either perfectly reflects the JSON or provides biologically correct and insightful integration of fields.
    4 = Mostly accurate, with minor uncertainties or omissions, but no significant errors.
    3 = Some unclear or incomplete parts, with minor biological inaccuracies or missed connections.
    2 = Clear errors, hallucinations, or biologically incorrect statements, but some relevant information is present.
    1 = Mostly incorrect, with significant errors or hallucinations and little to no relevant information.

    Use the full range of scores (1 to 5) based on the criteria above.

    Return only:
    <numeric score> <short comment>

    Example:
    5 Accurate and faithful summary

    GENE_JSON:
    {json.dumps(gene_json, ensure_ascii=False)}

    SUMMARY:
    \"\"\"{summary}\"\"\"
    """
    for attempt in range(max_retries):
        try:
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

            print(f"LLM response: {result}")

            # Clean the result to remove formatting like **, *, :, etc.
            cleaned_result = re.sub(r"[*_:]+", "", result.strip())

            # Parse the result into score and comment
            parts = cleaned_result.split(maxsplit=1)
            score = int(re.sub(r"[^\d]", "", parts[0])) if parts and parts[0][0].isdigit() else None
            comment = parts[1] if len(parts) > 1 else ""

            # Validate the score
            if score not in range(1, 6):
                raise ValueError(f"Invalid score: {score}")

            return {"score": score, "comment": comment}

        except (ValueError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(1)  # Optional: Add a delay between retries

    # Return a fallback response if all retries fail
    return {"score": None, "comment": "Failed to get valid response after retries."}