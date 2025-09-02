"""Manages cost estimation for medical tests in the DxO simulation."""
from functools import lru_cache
from typing import Dict, List, Union, Tuple, Set
from config import DEFAULT_TEST_COSTS


@lru_cache(maxsize=32)
def _keyword_sets(keys: Tuple[str, ...]) -> Dict[str, Set[str]]:
    """Precompute split keyword sets for cost database keys."""
    return {k: set(k.split()) for k in keys}

def estimate_cost(
    tests: Union[List[str], str], test_cost_db: Dict[str, int]
) -> int:
    """Estimate the cost of one or more diagnostic tests.

    Uses a simple keyword matching heuristic to find the most relevant
    cost entry in the provided database.

    Args:
        tests: A single test name or a list of test names.
        test_cost_db: A dictionary mapping test keywords to costs.

    Returns:
        The total estimated cost.
    """
    if isinstance(tests, str):
        tests = [tests]

    cost = 0
    # Precompute keyword sets for this database, caching by keys tuple
    keyword_sets = _keyword_sets(tuple(test_cost_db.keys()))

    for test in tests:
        test_lower = test.lower().strip()
        if not test_lower:
            continue

        test_words = set(test_lower.split())
        # Find the best matching key in the cost database
        best_match = max(
            keyword_sets.items(),
            key=lambda item: len(item[1] & test_words),
            default=("default", set()),
        )[0]
        cost += test_cost_db.get(best_match, test_cost_db.get("default", 150))
    return cost
