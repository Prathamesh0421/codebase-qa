# Multi-stage so build tooling does not ship. Note what is NOT installed here:
# the local-embeddings extra. The deployed image uses a hosted embedding API,
# which keeps torch (~2GB) out of the runtime layer. Local embeddings are a
# development and CI concern, where reproducibility matters more than size.

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

USER codeqa

EXPOSE 8000

# Replaced in Phase 14 by the uvicorn entrypoint, once there is an app to run.
CMD ["codeqa", "--help"]
