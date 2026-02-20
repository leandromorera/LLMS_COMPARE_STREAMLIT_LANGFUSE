"""Tests for src/vote.py — pure logic, no external dependencies."""
import pytest
from src.vote import majority_vote, vote_tally


class TestMajorityVote:
    def test_two_fake_wins(self):
        assert majority_vote(["FAKE", "FAKE", "REAL"]) == "FAKE"

    def test_two_real_wins(self):
        assert majority_vote(["REAL", "REAL", "FAKE"]) == "REAL"

    def test_all_fake(self):
        assert majority_vote(["FAKE", "FAKE", "FAKE"]) == "FAKE"

    def test_all_real(self):
        assert majority_vote(["REAL", "REAL", "REAL"]) == "REAL"

    def test_all_uncertain(self):
        assert majority_vote(["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"]) == "UNCERTAIN"

    def test_split_returns_uncertain(self):
        assert majority_vote(["FAKE", "REAL", "UNCERTAIN"]) == "UNCERTAIN"

    def test_two_uncertain_one_fake_returns_uncertain(self):
        assert majority_vote(["UNCERTAIN", "UNCERTAIN", "FAKE"]) == "UNCERTAIN"

    def test_two_fake_beats_one_uncertain(self):
        assert majority_vote(["FAKE", "FAKE", "UNCERTAIN"]) == "FAKE"


class TestVoteTally:
    def test_counts_correctly(self):
        tally = vote_tally(["FAKE", "REAL", "FAKE"])
        assert tally == {"FAKE": 2, "REAL": 1, "UNCERTAIN": 0}

    def test_all_uncertain(self):
        tally = vote_tally(["UNCERTAIN", "UNCERTAIN", "UNCERTAIN"])
        assert tally == {"FAKE": 0, "REAL": 0, "UNCERTAIN": 3}

    def test_empty_list(self):
        tally = vote_tally([])
        assert tally == {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0}

    def test_keys_always_present(self):
        tally = vote_tally(["REAL"])
        assert set(tally.keys()) == {"FAKE", "REAL", "UNCERTAIN"}
