"""ELO rating system for pairwise agent comparison.

Pure math — no I/O, no DB, no LLM.  Implements the standard ELO algorithm
used in chess and adapted by Chatbot Arena for LLM ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RATING = 1500.0
DEFAULT_K_FACTOR = 32.0


@dataclass
class EloResult:
    """Result of an ELO update for both players."""

    winner_new_rating: float
    loser_new_rating: float
    winner_delta: float
    loser_delta: float


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score of player A vs player B.

    Returns a value in (0, 1) representing the probability A wins.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    winner_rating: float,
    loser_rating: float,
    k_factor: float = DEFAULT_K_FACTOR,
    draw: bool = False,
) -> EloResult:
    """Compute new ELO ratings after a match.

    Args:
        winner_rating: Current rating of the winner (or player A in a draw).
        loser_rating: Current rating of the loser (or player B in a draw).
        k_factor: Maximum rating change per match.
        draw: If True, treat as a draw (0.5 / 0.5) instead of win/loss.
    """
    exp_winner = expected_score(winner_rating, loser_rating)
    exp_loser = 1.0 - exp_winner

    actual_winner = 0.5 if draw else 1.0
    actual_loser = 0.5 if draw else 0.0

    winner_delta = round(k_factor * (actual_winner - exp_winner), 2)
    loser_delta = round(k_factor * (actual_loser - exp_loser), 2)

    return EloResult(
        winner_new_rating=round(winner_rating + winner_delta, 2),
        loser_new_rating=round(loser_rating + loser_delta, 2),
        winner_delta=winner_delta,
        loser_delta=loser_delta,
    )


def compute_rankings(
    match_results: list[dict],
    initial_rating: float = DEFAULT_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
) -> dict[str, float]:
    """Compute ELO rankings from a chronological list of match results.

    Args:
        match_results: List of dicts with keys:
            - agent_config_id_a (str)
            - agent_config_id_b (str)
            - result: "a_wins" | "b_wins" | "draw"
        initial_rating: Starting rating for new agents.
        k_factor: K-factor for updates.

    Returns:
        Dict mapping agent_config_id → current ELO rating.
    """
    ratings: dict[str, float] = {}

    for match in match_results:
        a_id = match["agent_config_id_a"]
        b_id = match["agent_config_id_b"]
        result = match["result"]

        ratings.setdefault(a_id, initial_rating)
        ratings.setdefault(b_id, initial_rating)

        if result == "draw":
            elo = update_ratings(ratings[a_id], ratings[b_id], k_factor, draw=True)
            ratings[a_id] = elo.winner_new_rating
            ratings[b_id] = elo.loser_new_rating
        elif result == "a_wins":
            elo = update_ratings(ratings[a_id], ratings[b_id], k_factor)
            ratings[a_id] = elo.winner_new_rating
            ratings[b_id] = elo.loser_new_rating
        elif result == "b_wins":
            elo = update_ratings(ratings[b_id], ratings[a_id], k_factor)
            ratings[b_id] = elo.winner_new_rating
            ratings[a_id] = elo.loser_new_rating

    return ratings
