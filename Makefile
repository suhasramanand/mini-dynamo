.PHONY: help build up down restart logs ps smoke test clean fmt

COMPOSE ?= docker compose
ROUTER  ?= http://localhost:8000

help:
	@echo "Mini-Dynamo make targets:"
	@echo "  build     Build the service image"
	@echo "  up        Start the stack (redis, router, prefill, 2x decode)"
	@echo "  down      Stop the stack"
	@echo "  logs      Tail all logs"
	@echo "  ps        Show running services"
	@echo "  smoke     Run the smoke test against a running stack"
	@echo "  test      Run unit tests (needs local deps installed)"
	@echo "  clean     Stop stack and remove volumes"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --build
	@echo "Router:   $(ROUTER)"
	@echo "Backends: $(ROUTER)/backends"
	@echo "Metrics:  $(ROUTER)/metrics"

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

smoke:
	./scripts/smoke_test.sh $(ROUTER)

test:
	pytest -q

clean:
	$(COMPOSE) down -v

fmt:
	python -m pip install ruff >/dev/null 2>&1 || true
	ruff format common services benchmark tests || true
