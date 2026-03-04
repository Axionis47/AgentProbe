.PHONY: up down build logs test test-all lint migrate seed demo health clean

# ===== Infrastructure =====

up:
	docker compose up -d

up-build:
	docker compose up -d --build

down:
	docker compose down

down-clean:
	docker compose down -v --remove-orphans

build:
	docker compose build

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

logs-kafka-consumer:
	docker compose logs -f kafka-consumer

# ===== Database =====

migrate:
	docker compose exec api alembic upgrade head

migrate-down:
	docker compose exec api alembic downgrade -1

migrate-new:
	@read -p "Migration message: " msg; \
	docker compose exec api alembic revision --autogenerate -m "$$msg"

# ===== Testing =====

test:
	docker compose exec api pytest tests/unit -v

test-integration:
	docker compose exec api pytest tests/integration -v

test-e2e:
	docker compose exec api pytest tests/e2e -v

test-all:
	docker compose exec api pytest --cov=app --cov-report=term-missing -v

# ===== Quality =====

lint:
	docker compose exec api ruff check app/
	docker compose exec api mypy app/ --strict

format:
	docker compose exec api ruff format app/

# ===== Demo =====

seed:
	docker compose exec api python -m scripts.seed_data

demo:
	docker compose exec api python -m scripts.run_sample_eval

# ===== Health =====

health:
	@echo "Waiting for services to be healthy..."
	@until curl -sf http://localhost:8080/api/v1/health/ready > /dev/null 2>&1; do \
		echo "  API not ready, retrying in 3s..."; \
		sleep 3; \
	done
	@echo "API is healthy!"
	@until curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; do \
		echo "  Streamlit not ready, retrying in 3s..."; \
		sleep 3; \
	done
	@echo "Streamlit is healthy!"
	@until curl -sf http://localhost:3001/api/health > /dev/null 2>&1; do \
		echo "  Grafana not ready, retrying in 3s..."; \
		sleep 3; \
	done
	@echo "Grafana is healthy!"
	@echo "All services are ready!"

# ===== Cleanup =====

clean:
	docker compose down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
