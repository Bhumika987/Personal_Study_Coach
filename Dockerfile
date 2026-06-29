FROM python:3.12-slim

WORKDIR /app

# Install uv — fast dependency resolver
RUN pip install --no-cache-dir uv

# Copy lock files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all production dependencies into the project venv
RUN uv sync --frozen --no-dev

# Copy source (after deps so code changes don't bust the dep cache)
COPY . .

# Directories that need to exist at runtime
RUN mkdir -p temp_uploads faiss_index

# Default: run the FastAPI backend.
# docker-compose overrides this CMD for the Streamlit container.
EXPOSE 10000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
