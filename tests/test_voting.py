"""Tests for voting logic."""
import pytest
from src.voting import majority_vote, vote_tally, LABELS


class TestVoteTally:
    def test_all_fake(self):
        assert vote_tally(["FAKE", "FAKE", "FAKE"]) == {"FAKE": 3, "REAL": 0, "UNCERTAIN": 0}

    def test_all_real(self):
        assert vote_tally(["REAL", "REAL", "REAL"]) == {"FAKE": 0, "REAL": 3, "UNCERTAIN": 0}

    def test_all_uncertain(self):
        assert vote_tally(["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"]) == {"FAKE": 0, "REAL": 0, "UNCERTAIN": 3}

    def test_mixed_fake_real(self):
        tally = vote_tally(["FAKE", "REAL", "FAKE"])
        assert tally == {"FAKE": 2, "REAL": 1, "UNCERTAIN": 0}

    def test_empty(self):
        assert vote_tally([]) == {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0}

    def test_case_insensitive(self):
        tally = vote_tally(["fake", "FAKE", "real"])
        assert tally["FAKE"] == 2
        assert tally["REAL"] == 1

    def test_single_element(self):
        assert vote_tally(["FAKE"]) == {"FAKE": 1, "REAL": 0, "UNCERTAIN": 0}

    def test_unknown_label_ignored(self):
        tally = vote_tally(["FAKE", "INVALID", "REAL"])
        assert tally == {"FAKE": 1, "REAL": 1, "UNCERTAIN": 0}

    def test_two_uncertain(self):
        tally = vote_tally(["UNCERTAIN", "UNCERTAIN", "REAL"])
        assert tally["UNCERTAIN"] == 2


class TestMajorityVote:
    def test_unanimous_fake(self):
        assert majority_vote(["FAKE", "FAKE", "FAKE"]) == "FAKE"

    def test_unanimous_real(self):
        assert majority_vote(["REAL", "REAL", "REAL"]) == "REAL"

    def test_two_fake_one_real(self):
        assert majority_vote(["FAKE", "FAKE", "REAL"]) == "FAKE"

    def test_two_real_one_fake(self):
        assert majority_vote(["REAL", "REAL", "FAKE"]) == "REAL"

    def test_no_majority_is_uncertain(self):
        assert majority_vote(["FAKE", "REAL", "UNCERTAIN"]) == "UNCERTAIN"

    def test_fake_checked_before_real(self):
        # Both have 1 vote, should be UNCERTAIN
        assert majority_vote(["FAKE", "REAL", "UNCERTAIN"]) == "UNCERTAIN"

    def test_two_uncertain_is_uncertain(self):
        assert majority_vote(["UNCERTAIN", "UNCERTAIN", "REAL"]) == "UNCERTAIN"

    def test_labels_constant_contains_all(self):
        assert "FAKE" in LABELS
        assert "REAL" in LABELS
        assert "UNCERTAIN" in LABELS
