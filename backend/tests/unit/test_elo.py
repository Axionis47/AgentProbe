"""Tests for ELO rating system."""

from app.evaluation.elo import (
    DEFAULT_RATING,
    EloResult,
    compute_rankings,
    expected_score,
    update_ratings,
)


class TestExpectedScore:
    def test_equal_ratings_gives_half(self):
        assert expected_score(1500, 1500) == 0.5

    def test_higher_rating_expects_more(self):
        score = expected_score(1600, 1400)
        assert score > 0.5

    def test_lower_rating_expects_less(self):
        score = expected_score(1400, 1600)
        assert score < 0.5

    def test_symmetric(self):
        a = expected_score(1600, 1400)
        b = expected_score(1400, 1600)
        assert round(a + b, 6) == 1.0


class TestUpdateRatings:
    def test_winner_gains_loser_loses(self):
        result = update_ratings(1500, 1500)
        assert result.winner_new_rating > 1500
        assert result.loser_new_rating < 1500

    def test_deltas_are_symmetric(self):
        result = update_ratings(1500, 1500)
        assert result.winner_delta == -result.loser_delta

    def test_upset_gives_larger_delta(self):
        """Weaker player beating stronger gets bigger rating change."""
        normal = update_ratings(1600, 1400)  # expected wins
        upset = update_ratings(1400, 1600)   # upset
        assert upset.winner_delta > normal.winner_delta

    def test_draw_moves_toward_center(self):
        result = update_ratings(1600, 1400, draw=True)
        # Higher-rated player should lose points in a draw
        assert result.winner_new_rating < 1600
        # Lower-rated player should gain points
        assert result.loser_new_rating > 1400

    def test_returns_elo_result(self):
        result = update_ratings(1500, 1500)
        assert isinstance(result, EloResult)


class TestComputeRankings:
    def test_empty_matches(self):
        assert compute_rankings([]) == {}

    def test_single_match(self):
        matches = [{"agent_config_id_a": "a", "agent_config_id_b": "b", "result": "a_wins"}]
        ratings = compute_rankings(matches)
        assert ratings["a"] > DEFAULT_RATING
        assert ratings["b"] < DEFAULT_RATING

    def test_multiple_matches(self):
        matches = [
            {"agent_config_id_a": "a", "agent_config_id_b": "b", "result": "a_wins"},
            {"agent_config_id_a": "b", "agent_config_id_b": "c", "result": "b_wins"},
            {"agent_config_id_a": "a", "agent_config_id_b": "c", "result": "a_wins"},
        ]
        ratings = compute_rankings(matches)
        assert len(ratings) == 3
        # A won all, should be highest
        assert ratings["a"] > ratings["b"]
        assert ratings["a"] > ratings["c"]

    def test_draw_result(self):
        matches = [{"agent_config_id_a": "x", "agent_config_id_b": "y", "result": "draw"}]
        ratings = compute_rankings(matches)
        assert ratings["x"] == ratings["y"] == DEFAULT_RATING

    def test_b_wins_result(self):
        matches = [{"agent_config_id_a": "a", "agent_config_id_b": "b", "result": "b_wins"}]
        ratings = compute_rankings(matches)
        assert ratings["b"] > ratings["a"]
