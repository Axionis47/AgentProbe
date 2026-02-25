"""Tests for reference-based evaluator."""

import pytest

from app.evaluation.reference_evaluator import ReferenceEvaluator


class TestTokenOverlap:
    def test_identical_strings(self):
        assert ReferenceEvaluator._token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert ReferenceEvaluator._token_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = ReferenceEvaluator._token_overlap("the cat sat", "the dog sat on")
        assert 0 < score < 1

    def test_empty_actual(self):
        assert ReferenceEvaluator._token_overlap("", "hello") == 0.0

    def test_empty_expected(self):
        assert ReferenceEvaluator._token_overlap("hello", "") == 0.0


class TestLCSRatio:
    def test_identical_strings(self):
        assert ReferenceEvaluator._lcs_ratio("the cat sat", "the cat sat") == 1.0

    def test_partial_match(self):
        score = ReferenceEvaluator._lcs_ratio("a b c d", "a c e")
        assert 0 < score < 1

    def test_no_match(self):
        assert ReferenceEvaluator._lcs_ratio("a b c", "x y z") == 0.0

    def test_empty_strings(self):
        assert ReferenceEvaluator._lcs_ratio("", "hello") == 0.0


class TestExactMatch:
    def test_exact_same(self):
        assert ReferenceEvaluator._exact_match("Hello World", "hello world") == 1.0

    def test_whitespace_normalized(self):
        assert ReferenceEvaluator._exact_match("  hello   world  ", "hello world") == 1.0

    def test_different(self):
        assert ReferenceEvaluator._exact_match("hello", "world") == 0.0


class TestExtractPairs:
    def test_extracts_pairs(self):
        turns = [
            {"role": "user", "content": "question", "expected_response": "answer"},
            {"role": "assistant", "content": "actual answer"},
        ]
        pairs = ReferenceEvaluator._extract_pairs(turns)
        assert len(pairs) == 1
        assert pairs[0] == ("actual answer", "answer")

    def test_no_expected_response(self):
        turns = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        assert ReferenceEvaluator._extract_pairs(turns) == []

    def test_multiple_pairs(self):
        turns = [
            {"role": "user", "content": "q1", "expected_response": "a1"},
            {"role": "assistant", "content": "actual1"},
            {"role": "user", "content": "q2", "expected_response": "a2"},
            {"role": "assistant", "content": "actual2"},
        ]
        pairs = ReferenceEvaluator._extract_pairs(turns)
        assert len(pairs) == 2


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_no_references(self):
        evaluator = ReferenceEvaluator()
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = await evaluator.evaluate(turns, [])
        assert result.overall_score == 0.0
        assert result.evaluator_type == "reference_based"

    @pytest.mark.asyncio
    async def test_perfect_match(self):
        evaluator = ReferenceEvaluator()
        turns = [
            {"role": "user", "content": "q", "expected_response": "the answer"},
            {"role": "assistant", "content": "the answer"},
        ]
        result = await evaluator.evaluate(turns, [])
        assert result.overall_score == 10.0
        assert result.scores["exact_match"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluator_type(self):
        evaluator = ReferenceEvaluator()
        turns = [
            {"role": "user", "content": "q", "expected_response": "x"},
            {"role": "assistant", "content": "y"},
        ]
        result = await evaluator.evaluate(turns, [])
        assert result.evaluator_type == "reference_based"
