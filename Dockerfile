# Multi-stage so build tooling does not ship. Note what is NOT installed here:
# the local-embeddings extra. sentence-transformers + even a CPU-only torch
# build peaks around 800MB RSS just loading the model and embedding a small
# batch (measured directly, both PyTorch and ONNX Runtime backends -- ONNX
# was only marginally smaller) -- comfortably over the 512MB free-tier RAM
# cap of the host this image is meant to run on. Hosted embeddings (Cohere,
# not Gemini -- see indexing/embeddings.py's HostedEmbedder docstring for
# why Gemini's free-tier embedding quota made it a dead end too) keep this
# image small and sidestep the memory ceiling entirely. Local stays the
# dev/CI/eval-harness path, where reproducibility matters more than
# footprint and there's no 512MB constraint.

FROM python:3.14-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir hatchling

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip wheel --no-cache-dir --wheel-dir /wheels .


FROM python:3.14-slim AS runtime

# git is a runtime dependency, not a build one: repo ingestion shells out to
# `git clone --depth 1` for user-supplied URLs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Cloning untrusted repositories as root is unnecessary risk. Parsing itself is
# safe -- tree-sitter reads without executing -- but the clone writes to disk.
RUN useradd --create-home --uid 10001 codeqa

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels codeqa \
    && rm -rf /wheels

# config.py's clone_workdir defaults to ./data/repos, relative to the
# process's cwd -- fine on a dev machine where the whole tree is
# user-owned, but /app here is root-owned (created before USER switches
# below), so the worker's git clone would fail with a permission error the
# moment it tried to create it (found against a real deploy). Pre-creating
# and chowning it is cheaper than moving the default elsewhere, and every
# clone under it is reconstructible scratch space anyway -- nothing here
# needs to survive a restart.
RUN mkdir -p /app/data/repos && chown -R codeqa:codeqa /app/data

USER codeqa

EXPOSE 8000

CMD ["uvicorn", "codeqa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
