#!/bin/bash
# Run Songmaker with the correct virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for venv
if [ -d "venv" ]; then
    PYTHON="venv/bin/python"
elif [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    PYTHON="venv/bin/python"
    echo "Installing dependencies..."
    $PYTHON -m pip install -e .
fi

# Run Streamlit
exec $PYTHON -m streamlit run src/ui/app.py "$@"
