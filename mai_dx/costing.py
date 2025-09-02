"""
Manages cost estimation for medical tests in the DxO simulation.
"""
from typing import Dict, List, Union
from config import DEFAULT_TEST_COSTS

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
    for test in tests:
        test_lower = test.lower().strip()
        if not test_lower:
            continue

        # Find the best matching key in the cost database
        best_match = max(
            test_cost_db.keys(),
            key=lambda k: len(set(k.split()) & set(test_lower.split())),
            default="default",
        )
        cost += test_cost_db.get(best_match, test_cost_db.get("default", 150))
    return cost