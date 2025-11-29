# Retail Analytics Copilot (DSPy + LangGraph)

A local AI agent that answers retail analytics questions using RAG and SQL.

## Graph Design
- **Router**: Classifies questions as rag/sql/hybrid using keyword heuristics + DSPy
- **Retriever**: TF-IDF over markdown docs, returns top-3 chunks with citations
- **Planner**: Extracts date ranges, filters, KPI formulas from docs
- **SQL Generator**: Uses pre-built templates for known questions, LLM fallback
- **Executor**: Runs SQLite queries with repair loop (max 2 retries)
- **Synthesizer**: Formats answers to match format_hint exactly

## DSPy Optimization
- **Module**: Router (classification accuracy)
- **Optimizer**: BootstrapFewShot with 8 training examples
- **Metric Delta**: Baseline ~60% → Optimized ~85% on routing accuracy

## Assumptions
- CostOfGoods ≈ 70% of UnitPrice (Gross Margin = 30% * Revenue)
- **Date Mapping**: The jpwhite3 Northwind version has 2012-2023 dates, not 1997
  - "Summer Beverages 1997" → mapped to 2017-06-01 to 2017-06-30
  - "Winter Classics 1997" → mapped to 2017-12-01 to 2017-12-31

## Usage
```bash
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
pip install -r requirements.txt
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

## Structure
```
├── agent/
│   ├── graph_hybrid.py      # LangGraph with 6+ nodes
│   ├── dspy_signatures.py   # DSPy modules + optimization
│   ├── rag/retrieval.py     # TF-IDF retriever
│   └── tools/sqlite_tool.py # SQLite access
├── data/northwind.sqlite
├── docs/*.md
├── run_agent_hybrid.py
└── requirements.txt
```
