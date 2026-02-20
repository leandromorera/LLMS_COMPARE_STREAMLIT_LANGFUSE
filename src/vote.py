from __future__ import annotations



def majority_vote(labels: list[str]) -> str:

    # labels are "FAKE" | "REAL" | "UNCERTAIN"

    fake = sum(1 for x in labels if x == "FAKE")

    real = sum(1 for x in labels if x == "REAL")

    uncertain = sum(1 for x in labels if x == "UNCERTAIN")



    if fake >= 2:

        return "FAKE"

    if real >= 2:

        return "REAL"

    return "UNCERTAIN"



def vote_tally(labels: list[str]) -> dict:

    return {

        "FAKE": sum(1 for x in labels if x == "FAKE"),

        "REAL": sum(1 for x in labels if x == "REAL"),

        "UNCERTAIN": sum(1 for x in labels if x == "UNCERTAIN"),

    }
