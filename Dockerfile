FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so they're cached separately from source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY README.md ./
RUN uv sync --frozen

COPY docker/torrc /etc/tor/torrc
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"
ENV TOR_SOCKS_PROXY="socks5h://127.0.0.1:9050"
ENV TOR_CONTROL_PORT="9051"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["darknet-scraper"]
