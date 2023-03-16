
from typing import Dict

def counts_to_ratios(counts: Dict[str, int]) -> Dict[str, float]:
    s = sum(counts.values())
    return {k: v / s for k, v in counts.items()}
