#!/bin/bash
uv run uvicorn main:app --host 0.0.0.0 --port 10000 &

API_BASE_URL=http://localhost:10000 \
uv run streamlit run streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
