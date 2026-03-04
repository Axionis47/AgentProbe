"""Abstract base consumer with retry logic, idempotency, and DLQ.

All concrete consumers inherit from this and implement handle_event().
"""

from __future__ import annotations

import abc
import threading
import time
from typing import Any

import structlog

from app.config import settings
from app.pipeline.events import EventEnvelope
from app.pipeline.topics import PIPELINE_ERRORS

logger = structlog.get_logger()


class BaseConsumer(abc.ABC):
    """Base Kafka consumer with idempotency and dead letter queue support."""

    def __init__(
        self,
        topic: str,
        group_id: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.topic = topic
        self.group_id = group_id or settings.kafka_consumer_group
        self.max_retries = max_retries

        self._running = False
        self._thread: threading.Thread | None = None
        self._processed_ids: set[str] = set()
        self._consumer: Any = None

    def start(self) -> None:
        """Subscribe and start consuming in a background thread."""
        from confluent_kafka import Consumer

        self._consumer = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "session.timeout.ms": 30000,
        })
        self._consumer.subscribe([self.topic])

        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop,
            name=f"consumer-{self.topic}",
            daemon=True,
        )
        self._thread.start()
        logger.info("consumer_started", topic=self.topic, group_id=self.group_id)

    def stop(self) -> None:
        """Signal shutdown, wait for thread, close consumer."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        if self._consumer:
            self._consumer.close()
        logger.info("consumer_stopped", topic=self.topic)

    def _consume_loop(self) -> None:
        """Poll → deserialize → idempotency check → process with retries."""
        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("consumer_poll_error", error=str(msg.error()), topic=self.topic)
                continue

            try:
                envelope = EventEnvelope.deserialize(msg.value())
            except Exception as e:
                logger.error("consumer_deserialize_error", error=str(e), topic=self.topic)
                continue

            # Idempotency: skip already-processed events
            event_id = envelope.payload.get("event_id")
            if event_id and event_id in self._processed_ids:
                logger.debug("consumer_duplicate_skipped", event_id=event_id, topic=self.topic)
                continue

            self._process_with_retries(envelope, msg)

    def _process_with_retries(self, envelope: EventEnvelope, msg: Any) -> None:
        """Retry processing up to max_retries times, then DLQ."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.handle_event(envelope)

                # Mark as processed after success
                event_id = envelope.payload.get("event_id")
                if event_id:
                    self._processed_ids.add(event_id)
                    # Cap set size to prevent unbounded memory growth
                    if len(self._processed_ids) > 100_000:
                        # Evict oldest half (approximation since sets are unordered)
                        to_remove = list(self._processed_ids)[:50_000]
                        self._processed_ids -= set(to_remove)

                return

            except Exception as e:
                logger.warning(
                    "consumer_retry",
                    topic=self.topic,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))  # Exponential backoff, max 30s

        # All retries exhausted → send to DLQ
        self._send_to_dlq(msg, "Max retries exhausted")

    def _send_to_dlq(self, msg: Any, error: str) -> None:
        """Produce failed message to the pipeline.errors topic."""
        try:
            from app.pipeline.producer import KafkaProducer

            dlq_envelope = EventEnvelope(
                version=1,
                event_type="pipeline.dead_letter",
                payload={
                    "original_topic": self.topic,
                    "error": error,
                    "original_value": msg.value().decode("utf-8") if msg.value() else "",
                },
            )
            producer = KafkaProducer()
            producer.produce(PIPELINE_ERRORS, dlq_envelope)
            logger.error("consumer_dlq", topic=self.topic, error=error)
        except Exception as dlq_err:
            logger.error("consumer_dlq_failed", topic=self.topic, error=str(dlq_err))

    @abc.abstractmethod
    def handle_event(self, envelope: EventEnvelope) -> None:
        """Process a single event. Implemented by concrete consumers."""
        ...
