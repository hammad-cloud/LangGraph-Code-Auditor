@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo Starting LangGraph code auditor...
echo Open http://localhost:8504 in your browser
.venv\Scripts\streamlit run app.py --server.headless true --server.port 8504 --server.address localhost
