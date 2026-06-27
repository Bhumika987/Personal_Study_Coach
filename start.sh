#!/bin/bash
echo "Starting Personal Study Coach API..."
uvicorn main:app --host 0.0.0.0 --port "${PORT:-10000}"
