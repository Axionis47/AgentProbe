"""Unit tests for the base Kafka consumer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.pipeline.consumers.base import BaseConsumer
from app.pipeline.events import EventEnvelope


class ConcreteConsumer(BaseConsumer):
    """Test-only concrete implementation."""

    def __init__(self) -> None:
        super().__init__(topic="test.topic", max_retries=3)
        self.received_events: list[EventEnvelope] = []

    def handle_event(self, envelope: EventEnvelope) -> None:
        self.received_events.append(envelope)


class FailingConsumer(BaseConsumer):
    """Consumer that always raises."""

    def __init__(self) -> None:
        super().__init__(topic="test.topic", max_retries=2)
        self.attempts = 0

    def handle_event(self, envelope: EventEnvelope) -> None:
        self.attempts += 1
        raise RuntimeError("Processing failed")


class TestBaseConsumerIdempotency:

    def test_duplicate_event_skipped(self) -> None:
        """Same event_id should only be processed once."""
        consumer = ConcreteConsumer()

        envelope = EventEnvelope(
            version=1,
            event_type="test.event",
            payload={"event_id": "evt-123", "data": "hello"},
        )

        mock_msg = MagicMock()
        mock_msg.value.return_value = envelope.serialize()

        # Process first time
        consumer._process_with_retries(envelope, mock_msg)
        assert len(consumer.received_events) == 1

        # Second time — should be skipped by idempotency
        assert "evt-123" in consumer._processed_ids

    def test_event_without_id_still_processed(self) -> None:
        """Events without event_id should still be processed."""
        consumer = ConcreteConsumer()

        envelope = EventEnvelope(
            version=1,
            event_type="test.event",
            payload={"data": "no id"},
        )
        mock_msg = MagicMock()
        consumer._process_with_retries(envelope, mock_msg)
        assert len(consumer.received_events) == 1


class TestBaseConsumerRetries:

    @patch("app.pipeline.consumers.base.time.sleep")
    def test_retries_exhaust_then_dlq(self, mock_sleep: MagicMock) -> None:
        """After max_retries, message should be sent to DLQ."""
        consumer = FailingConsumer()

        envelope = EventEnvelope(
            version=1,
            event_type="test.event",
            payload={"event_id": "evt-fail"},
        )
        mock_msg = MagicMock()
        mock_msg.value.return_value = envelope.serialize()

        with patch.object(consumer, "_send_to_dlq") as mock_dlq:
            consumer._process_with_retries(envelope, mock_msg)

            assert consumer.attempts == 2  # max_retries = 2
            mock_dlq.assert_called_once()
            assert "Max retries exhausted" in mock_dlq.call_args[0][1]


class TestBaseConsumerHandleEvent:

    def test_concrete_consumer_receives_envelope(self) -> None:
        """Concrete consumer's handle_event receives the correct envelope."""
        consumer = ConcreteConsumer()

        envelope = EventEnvelope(
            version=1,
            event_type="test.event",
            payload={"event_id": "evt-456", "key": "value"},
        )
        mock_msg = MagicMock()
        consumer._process_with_retries(envelope, mock_msg)

        assert len(consumer.received_events) == 1
        assert consumer.received_events[0].payload["key"] == "value"
