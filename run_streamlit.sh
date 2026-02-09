#!/bin/bash
# Run Streamlit app for Zero-Shot WorldCoder
# Run from project root: ./run_streamlit.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Streamlit app..."
echo "Open http://localhost:8501 in your browser"

streamlit run app/streamlit_app.py --server.port 8501





