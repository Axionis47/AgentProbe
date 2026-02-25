"""Thread-safe Kafka producer singleton.

All event publishing goes through this single producer instance.
Uses confluent-kafka with idempotent production for exactly-once semantics.
"""

from __future__ import annotations

import threading
from typing import Any

import structlog

from app.config import settings
from app.pipeline.events import EventEnvelope

logger = structlog.get_logger()


class KafkaProducer:
    """Singleton Kafka producer with thread-safe initialization."""

    _instance: KafkaProducer | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> KafkaProducer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        from confluent_kafka import Producer

        self._producer = Producer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "acks": "all",
            "retries": 3,
            "enable.idempotence": True,
            "max.in.flight.requests.per.connection": 5,
            "linger.ms": 10,
        })
        self._initialized = True
        logger.info("kafka_producer_initialized", bootstrap=settings.kafka_bootstrap_servers)

    def produce(
        self,
        topic: str,
        envelope: EventEnvelope,
        key: str | None = None,
    ) -> None:
        """Serialize and publish an event envelope to a Kafka topic."""
        data = envelope.serialize()
        self._producer.produce(
            topic=topic,
            value=data,
            key=key.encode("utf-8") if key else None,
            callback=self._delivery_callback,
        )
        # Trigger any queued delivery callbacks
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> int:
        """Flush pending messages. Returns number of messages still in queue."""
        return self._producer.flush(timeout=timeout)

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — used in tests."""
        with cls._lock:
            if cls._instance is not None and cls._instance._initialized:
                try:
                    cls._instance._producer.flush(timeout=2.0)
                except Exception:
                    pass
            cls._instance = None

    @staticmethod
    def _delivery_callback(err: Any, msg: Any) -> None:
        """Called once per message to indicate delivery result."""
        if err is not None:
            logger.error(
                "kafka_delivery_failed",
                error=str(err),
                topic=msg.topic() if msg else "unknown",
            )
        else:
            logger.debug(
                "kafka_delivery_success",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )
