"""Unit tests for AdversarialStrategy."""

from __future__ import annotations

from app.engine.adversarial import (
    ALL_ADVERSARIAL,
    AdversarialStrategy,
    NoOpAdversarial,
)
from app.engine.environment import SimulationEnvironment


class TestAdversarialStrategy:
    def test_should_inject_at_configured_turns(self) -> None:
        env = SimulationEnvironment(adversarial_turns=[2, 4])
        strategy = AdversarialStrategy(env)

        assert not strategy.should_inject(0)
        assert not strategy.should_inject(1)
        assert strategy.should_inject(2)
        assert not strategy.should_inject(3)
        assert strategy.should_inject(4)

    def test_generate_returns_nonempty_string(self) -> None:
        env = SimulationEnvironment(adversarial_turns=[0])
        strategy = AdversarialStrategy(env)

        msg = strategy.generate_adversarial_input(0)
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_messages_come_from_known_categories(self) -> None:
        env = SimulationEnvironment(adversarial_turns=[0])
        strategy = AdversarialStrategy(env)

        all_messages = set()
        for category in ALL_ADVERSARIAL:
            all_messages.update(category)

        # Generate 50 messages, all should be from known categories
        for _ in range(50):
            msg = strategy.generate_adversarial_input(0)
            assert msg in all_messages


class TestNoOpAdversarial:
    def test_never_injects(self) -> None:
        noop = NoOpAdversarial()

        for i in range(100):
            assert not noop.should_inject(i)

    def test_returns_empty_string(self) -> None:
        noop = NoOpAdversarial()
        assert noop.generate_adversarial_input(0) == ""
