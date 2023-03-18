
from typing import Dict

def counts_to_ratios(counts: Dict[str, int]) -> Dict[str, float]:
    """Converts a dictionary of counts to one of ratios (or proportions) between 0 and 1."""
    s = sum(counts.values())
    return {k: v / s for k, v in counts.items()}
