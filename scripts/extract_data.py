import requests
from bs4 import BeautifulSoup

def fetch_gene_json(ensembl_id):
    """Fetch the main gene JSON from Human Protein Atlas."""
    url = f"https://www.proteinatlas.org/{ensembl_id}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_blood_info_from_html(ensembl_id, gene_symbol):
    """
    Fetch the blood HTML and extract HUMAN PROTEIN ATLAS INFORMATION table.
    Returns a dictionary of key-value pairs from the table, only for required keys.
    """

    url = f"https://www.proteinatlas.org/{ensembl_id}-{gene_symbol}/blood"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    atlas_info = {}
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            if not cols or len(cols) < 2:
                continue
            if "Upregulated in disease" in cols[0]:
                atlas_info["Upregulated in disease"] = cols[1]
            if "Blood-based immunoassay" in cols[0]:
                atlas_info["Blood-based immunoassay"] = cols[1]
            if "Mass spectrometry" in cols[0]:
                atlas_info["Mass spectrometry"] = cols[1]
            if "Proximity extension assay" in cols[0]:
                atlas_info["Proximity extension assay"] = cols[1]
    return atlas_info

def extract_brain_info_from_json(data):
    """Extract only the structured brain info block from the main JSON (no general tissue fields)."""
    return {
        "Regional specificity": {
            "Human brain": data.get("RNA brain regional specificity"),
            "Pig brain": data.get("RNA pig brain regional specificity"),
            "Mouse brain": data.get("RNA mouse brain regional specificity"),
        },
        "Tau specificity score": {
            "Human brain": data.get("RNA brain regional specificity score"),
            "Pig brain": data.get("RNA pig brain regional specificity score"),
            "Mouse brain": data.get("RNA mouse brain regional specificity score"),
        },
        "Regional distribution": {
            "Human brain": data.get("RNA brain regional distribution"),
            "Pig brain": data.get("RNA pig brain regional distribution"),
            "Mouse brain": data.get("RNA mouse brain regional distribution"),
        }
    }

def merge_dicts_no_duplicates(*dicts):
    """Merge dictionaries, ensuring no duplicate keys."""
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result:
                raise ValueError(f"Duplicate key found during merge: {k}")
            result[k] = v
    return result

def extract_gene_summary(ensembl_id):
    """Extract and combine all relevant information for a single gene."""
    gene_json = fetch_gene_json(ensembl_id)
    gene_symbol = gene_json.get("Gene")
    if not gene_symbol:
        raise ValueError(f"Gene symbol not found in JSON for {ensembl_id}")

    # Extract brain info from JSON
    brain_info = extract_brain_info_from_json(gene_json)
    # Extract blood info from HTML
    blood_info = fetch_blood_info_from_html(ensembl_id, gene_symbol)

    # Merge all info, checking for duplicates
    combined = merge_dicts_no_duplicates(
        gene_json,
        {"HUMAN PROTEIN ATLAS INFORMATION (brain)": brain_info},
        {"HUMAN PROTEIN ATLAS INFORMATION (blood)": blood_info}
    )
    return combined

def extract_multiple_genes(ensembl_ids):
    """Extract summaries for multiple genes, return as dict {ENSG: summary_dict}."""
    results = {}
    for ensg in ensembl_ids:
        try:
            results[ensg] = extract_gene_summary(ensg)
        except Exception as e:
            print(f"Error processing {ensg}: {e}")
    return results