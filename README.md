# LangGraph Sequential Code Auditor

AI-powered development tool that scans, refactors, and documents Python source code through a sequential multi-agent pipeline powered by **LangGraph** and **Google Gemini** (`gemini-3.6-flash`).

## Architecture

```
Scanner Agent  →  Refactor Agent  →  Docs Agent
   (audit)         (clean code)      (documentation)
```

| Agent | Role |
|-------|------|
| **Scanner** | Detects SQL injection, security flaws, bugs, performance issues, and PEP 8 violations |
| **Refactor** | Produces optimized, secure, PEP 8-compliant Python code |
| **Docs** | Generates function docstrings and a modular README block |

## Quick Start

### 1. Clone & install

```bash
cd langgraph-code-auditor
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Project Structure

```
langgraph-code-auditor/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                  # Streamlit frontend
└── src/
    ├── __init__.py
    ├── state.py            # AgentState TypedDict
    ├── agents.py           # Scanner, Refactor, Docs agents
    ├── graph.py            # LangGraph StateGraph pipeline
    └── utils.py            # LLM init & response parsing
```

## Usage

1. Paste Python code or upload a `.py` file.
2. Enter your Google API key in the sidebar (or set it in `.env`).
3. Click **Run Audit Pipeline**.
4. Watch real-time progress as each agent completes.
5. Review results in the **Audit Report**, **Clean Code**, and **Documentation** tabs.

## Tech Stack

- **Python 3.10+**
- **LangGraph** — sequential agent orchestration
- **LangChain + langchain-google-genai** — Gemini LLM integration
- **Streamlit** — interactive dashboard
- **TypedDict** — deterministic state management

## License

MIT
