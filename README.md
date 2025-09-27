# Mini Agentic Pipeline

## Setup
1. Clone repo
2. Create virtual environment
   python -m venv venv
3. Activate venv
   venv\Scripts\activate
4. Install dependencies
   pip install pandas scikit-learn numpy

## Run Demo
python mini_agent.py

## Design Decisions
- Dummy rule-based Reasoner (no API key required)
- CSV used as tool for price lookup
- TF-IDF retriever with 10+ docs
- JSON logs include latency per step

## Known Limitations
- Rule-based Reasoner, no real LLM
- Simple product name extraction from query
- Retrieval uses TF-IDF, not semantic embeddings
