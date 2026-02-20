"""
Majority vote aggregation for 3-judge panel.
FAKE or REAL needs ≥2/3 votes; otherwise UNCERTAIN.
FAKE is checked before REAL (conservative).
"""
LABELS = {"FAKE", "REAL", "UNCERTAIN"}


def majority_vote(labels: list[str]) -> str:
    """Return FAKE, REAL, or UNCERTAIN based on majority of 3 labels."""
    tally = vote_tally(labels)
    if tally.get("FAKE", 0) >= 2:
        return "FAKE"
    if tally.get("REAL", 0) >= 2:
        return "REAL"
    return "UNCERTAIN"


def vote_tally(labels: list[str]) -> dict:
    """Return {FAKE: n, REAL: n, UNCERTAIN: n} counts."""
    result = {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0}
    for label in labels:
        upper = label.upper()
        if upper in result:
            result[upper] += 1
    return result
