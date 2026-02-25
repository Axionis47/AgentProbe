"""Entry point for running all Kafka consumers.

Used by docker-compose kafka-consumer service:
    python -m app.pipeline.consumers.run_all
"""

from __future__ import annotations

import signal
import sys
import time

import structlog

from app.pipeline.consumers.conversation_consumer import ConversationCompletedConsumer
from app.pipeline.consumers.evaluation_consumer import EvaluationCompletedConsumer
from app.pipeline.consumers.metrics_consumer import MetricsAggregatedConsumer

logger = structlog.get_logger()


def main() -> None:
    """Start all consumers and block until SIGTERM/SIGINT."""
    consumers = [
        ConversationCompletedConsumer(),
        EvaluationCompletedConsumer(),
        MetricsAggregatedConsumer(),
    ]

    def shutdown(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("shutdown_signal_received", signal=sig_name)
        for consumer in consumers:
            consumer.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info("starting_all_consumers", count=len(consumers))
    for consumer in consumers:
        consumer.start()

    # Block main thread
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        for consumer in consumers:
            consumer.stop()


if __name__ == "__main__":
    main()
