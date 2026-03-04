"""Unit tests for the Kafka producer singleton."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.pipeline.events import ConversationCompletedEvent, EventEnvelope
from app.pipeline.producer import KafkaProducer
from app.pipeline.topics import CONVERSATION_COMPLETED


class TestKafkaProducer:

    def setup_method(self) -> None:
        KafkaProducer.reset()

    def teardown_method(self) -> None:
        KafkaProducer.reset()

    @patch("confluent_kafka.Producer")
    def test_singleton_creates_one_instance(self, mock_producer_cls: MagicMock) -> None:
        """Two instantiations should return the same object."""
        p1 = KafkaProducer()
        p2 = KafkaProducer()
        assert p1 is p2
        # confluent_kafka.Producer should only be called once
        assert mock_producer_cls.call_count == 1

    @patch("confluent_kafka.Producer")
    def test_produce_serializes_and_sends(self, mock_producer_cls: MagicMock) -> None:
        """produce() should serialize the envelope and call the underlying producer."""
        mock_inner = MagicMock()
        mock_producer_cls.return_value = mock_inner

        producer = KafkaProducer()
        event = ConversationCompletedEvent(
            eval_run_id="run-1",
            conversation_id="conv-1",
            turn_count=5,
            total_tokens=100,
            total_latency_ms=500,
            status="completed",
        )
        envelope = event.to_envelope()

        producer.produce(CONVERSATION_COMPLETED, envelope, key="conv-1")

        mock_inner.produce.assert_called_once()
        call_kwargs = mock_inner.produce.call_args
        assert call_kwargs.kwargs["topic"] == CONVERSATION_COMPLETED
        assert call_kwargs.kwargs["key"] == b"conv-1"
        assert isinstance(call_kwargs.kwargs["value"], bytes)

    @patch("confluent_kafka.Producer")
    def test_flush_delegates_to_inner(self, mock_producer_cls: MagicMock) -> None:
        mock_inner = MagicMock()
        mock_inner.flush.return_value = 0
        mock_producer_cls.return_value = mock_inner

        producer = KafkaProducer()
        result = producer.flush(timeout=5.0)

        mock_inner.flush.assert_called_once_with(timeout=5.0)
        assert result == 0
