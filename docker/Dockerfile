FROM python:3.11-slim AS python-base

# Python configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    VENV_PATH="/app/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Common base stage with shared dependencies
FROM python-base AS base
ENV BUILD_ENV=base

# Install common runtime dependencies and create user first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    # Setup user
    groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 appuser && \
    # Create app directories with correct permissions
    mkdir -p /app /app/data /app/models /app/logs && \
    chown -R appuser:appuser /app /opt

# Set working directory
WORKDIR /app

# Switch to appuser for Poetry installation
USER appuser

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.in-project true

# Copy project files with correct ownership
COPY --chown=appuser:appuser pyproject.toml poetry.lock* ./

# Development stage
FROM base AS dev
ENV BUILD_ENV=dev

USER root
# Install additional development dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy and setup entrypoint
COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

USER appuser

# Set entrypoint and default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]

# Production stage
FROM base AS prod
ENV BUILD_ENV=prod

USER root
# Copy and setup entrypoint
COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

USER appuser

# Set entrypoint and default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
