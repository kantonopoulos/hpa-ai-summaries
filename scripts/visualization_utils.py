def flatten_values(val):
    """Recursively flatten lists/dicts to final values."""
    if isinstance(val, dict):
        vals = []
        for v in val.values():
            vals.extend(flatten_values(v))
        return vals
    elif isinstance(val, list):
        vals = []
        for v in val:
            vals.extend(flatten_values(v))
        return vals
    else:
        return [val]

def get_group(field):
    """Group fields by prefix."""
    if field.startswith("RNA"):
        return "RNA"
    elif field.startswith("Cancer prognostics"):
        return "Cancer prognostics"
    elif field.startswith("Blood"):
        return "Blood"
    elif field.startswith("Brain"):
        return "Brain"
    elif field.startswith("Antibody"):
        return "Antibody"
    elif field.startswith("Reliability"):
        return "Reliability"
    elif field.startswith("Subcellular"):
        return "Subcellular"
    elif field.startswith("Secretome"):
        return "Secretome"
    elif field.startswith("CCD"):
        return "CCD"
    elif field.startswith("Protein class"):
        return "Protein class"
    elif field.startswith("Interactions"):
        return "Interactions"
    else:
        return "Other"