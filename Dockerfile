FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

RUN mkdir -p temp_uploads faiss_index && chmod +x start.sh

EXPOSE 7860 10000

CMD ["./start.sh"]
