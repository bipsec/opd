# qopd — every target runs inside the dev container unless we are already in one.
IMAGE    ?= qopd-dev:26.10
BASE     ?= nvcr.io/nvidia/vllm:26.10-py3
HF_CACHE ?= $(HOME)/.cache/huggingface
SRC      := src tests scripts docker

DOCKER_FLAGS = --rm --gpus all --ipc=host \
	-u $(shell id -u):$(shell id -g) -e HOME=/tmp/home \
	-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
	-v $(CURDIR):/workspace/opd -v $(HF_CACHE):/hf -w /workspace/opd

ifeq ($(wildcard /.dockerenv),)
IN := docker run $(DOCKER_FLAGS) $(IMAGE)
else
IN :=
endif

.PHONY: help dev-image shell lint format test test-gpu lock env

help:
	@echo "dev-image  build $(IMAGE) from $(BASE) + requirements/dev.txt"
	@echo "shell      interactive shell in the dev container"
	@echo "lint       ruff check + format check"
	@echo "format     ruff format + autofix"
	@echo "test       fast CPU tests (not gpu, not slow)"
	@echo "test-gpu   gpu and slow tests"
	@echo "lock       pip freeze > requirements/lock.txt (inside the container)"
	@echo "env        qopd env (environment report)"

dev-image:
	docker build --build-arg BASE=$(BASE) -f docker/Dockerfile.dev -t $(IMAGE) .

shell:
	docker run -it $(DOCKER_FLAGS) $(IMAGE) bash

lint:
	$(IN) sh -c "ruff check $(SRC) && ruff format --check $(SRC)"

format:
	$(IN) sh -c "ruff check --fix $(SRC) && ruff format $(SRC)"

test:
	$(IN) python -m pytest -m "not gpu and not slow"

test-gpu:
	$(IN) python -m pytest -m "gpu or slow"

lock:
	$(IN) sh -c "pip freeze --exclude-editable > requirements/lock.txt"

env:
	$(IN) qopd env
