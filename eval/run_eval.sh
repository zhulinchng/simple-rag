#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

cp eval/eval_queries.json queries.json
make clean

python run_pipeline.py

echo ""
echo "--- Retrieval Metrics ---"
cat retrieval_metrics.json
