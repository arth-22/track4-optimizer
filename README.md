# Track 4: Cost-Quality Optimization via Historical Replay

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Portkey](https://img.shields.io/badge/Portkey-Model%20Catalog-purple.svg)](https://portkey.ai/)

> **Hackathon Project**: Portkey AI Builders Challenge - Track 4  
> Analyze historical LLM usage, replay through alternative models, and get data-driven cost-quality optimization recommendations.

---

## 🎯 What This Does

```
Historical Prompts → Replay Through 5 Models → Evaluate Quality → Recommend Optimal Routing
```

**Input**: Your existing prompts from Portkey logs or CSV files  
**Output**: Actionable recommendation like *"Switch simple prompts to Gemini Flash for 96% cost savings with minimal quality impact"*

---

## 🚀 Quick Start (5 Minutes)

### 1. Clone & Install

```bash
git clone https://github.com/arth-22/track4-optimizer.git
cd track4-optimizer

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Portkey API Key

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your Portkey API key
PORTKEY_API_KEY=your-portkey-api-key-here
```

### 3. Verify Connection

```bash
python3 scripts/test_portkey.py
```

Expected output:
```
✅ @openai/gpt-4o: Hello!
✅ @openai/gpt-4o-mini: Hello!
✅ @anthropic/claude-sonnet-4-5: Hello!
✅ @anthropic/claude-haiku-4-5-20251001: Hello!
✅ @vertex/gemini-2.5-flash: Hello

Result: 5/5 models working
🎉 All models ready for hackathon!
```

### 4. Run Demo

```bash
python3 scripts/demo.py --prompts 100
```

This will:
1. Generate 100 synthetic prompts
2. Simulate replay across 5 models
3. Calculate quality scores and costs
4. Generate `demo_report.html` with interactive Pareto chart

**Open the report:**
```bash
open demo_report.html  # macOS
# or just open in browser
```

---

## 📁 Project Structure

```
track4-optimizer/
├── src/
│   ├── adapters/          # Data ingestion (Portkey logs, CSV/JSON)
│   ├── replay/            # Replay engine with Model Catalog
│   ├── evaluation/        # Quality metrics (BERTScore, DeepEval, Guardrails)
│   ├── analysis/          # Statistics, Pareto, Anomaly Detection
│   ├── api/               # FastAPI endpoints
│   ├── models/            # Pydantic data models
│   └── db/                # SQLAlchemy persistence (optional)
├── scripts/
│   ├── demo.py            # Demo script
│   ├── production_demo.py # Production demo with real Portkey data
│   ├── test_portkey.py    # Connection test
│   └── generate_synthetic_data.py
├── tests/                 # 29 unit/integration tests
├── requirements.txt
└── .env.example
```

---

## 🔥 Production Demo (Real Data)

Run the full pipeline with **real Portkey logs**:

```bash
# Standard run (50 prompts, 5 models)
python3 scripts/production_demo.py --source portkey --limit 50 -y

# Quick mode (faster, skips BERTScore)
python3 scripts/production_demo.py --source portkey --limit 20 --quick -y

# Daemon mode (continuous execution every 6 hours)
python3 scripts/production_demo.py --source portkey --limit 50 --daemon --interval 6 -y
```

### Production Features

| Feature | Description |
|---------|-------------|
| **Daemon Mode** | `--daemon` runs continuously at `--interval` hours |
| **Real Portkey Logs** | Uses Log Export API (create → start → poll → download) |
| **Guardrails** | Toxicity, PII leakage, prompt injection detection |
| **Anomaly Detection** | LLM-powered pattern analysis |
| **Graceful Degradation** | Fallback evaluators when primary fails |
| **LLM Segmentation** | AI-driven prompt categorization |

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Required
PORTKEY_API_KEY=your-portkey-api-key

# Optional: Provider slugs (defaults shown)
OPENAI_PROVIDER_SLUG=openai
ANTHROPIC_PROVIDER_SLUG=anthropic
GOOGLE_PROVIDER_SLUG=vertex

# Optional: For DeepEval judge model
OPENAI_API_KEY=sk-xxx  # Only needed for LLM-as-judge evaluation
```

### Available Models

The system is pre-configured with 5 models via Portkey's Model Catalog:

| Model | Slug | Cost (per 1M tokens) |
|-------|------|----------------------|
| GPT-4o | `@openai/gpt-4o` | $2.50 / $10.00 |
| GPT-4o Mini | `@openai/gpt-4o-mini` | $0.15 / $0.60 |
| Claude Sonnet 4.5 | `@anthropic/claude-sonnet-4-5` | $3.00 / $15.00 |
| Claude Haiku 4.5 | `@anthropic/claude-haiku-4-5-20251001` | $0.80 / $4.00 |
| Gemini 2.5 Flash | `@vertex/gemini-2.5-flash` | $0.10 / $0.40 |

---

## 📊 Usage Examples

### Example 1: Run Demo with Simulated Data

```bash
# Quick demo (50 prompts, simulated)
python3 scripts/demo.py --prompts 50

# Larger demo (500 prompts)
python3 scripts/demo.py --prompts 500
```

### Example 2: Run Demo with Live API Calls

```bash
# Uses real Portkey API (slower, costs money)
python3 scripts/demo.py --prompts 20 --live
```

### Example 3: Start the API Server

```bash
# Start FastAPI server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Open API docs
open http://localhost:8000/docs
```

### Example 4: Analyze Your Own Data

```python
import asyncio
from src.adapters.csv_adapter import CSVAdapter
from src.replay.engine import ReplayEngine
from src.replay.model_registry import ModelRegistry
from src.analysis.recommendation import RecommendationEngine

async def analyze_my_data():
    # 1. Load your prompts
    adapter = CSVAdapter("path/to/your/prompts.json")
    prompts = await adapter.fetch_prompts(limit=100)
    
    # 2. Replay through models
    engine = ReplayEngine()
    results = []
    async for result in engine.replay_batch(prompts):
        results.append(result)
    
    # 3. Generate recommendation
    rec_engine = RecommendationEngine()
    recommendation = rec_engine.generate_recommendation(
        prompts=prompts,
        evaluations=results,
        current_model="gpt-4o"
    )
    
    print(f"Cost savings: {recommendation.total_cost_reduction_percent}%")
    print(recommendation.executive_summary)

asyncio.run(analyze_my_data())
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: API & Presentation                       │
│  FastAPI Endpoints  │  HTML Reports  │  Plotly Visualizations       │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 3: Analysis Engine                          │
│  Statistics (CI)  │  Segmentation  │  Pareto Frontier  │  Recommend │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 2: Evaluation Framework                     │
│  BERTScore  │  DeepEval (LLM Judge)  │  Cost Tracker  │  Composite  │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: Data & Replay                            │
│  Portkey Adapter  │  CSV Adapter  │  Replay Engine  │  Model Registry│
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PORTKEY GATEWAY                               │
│  Model Catalog (@provider/model)  │  Unified API  │  Rate Limiting  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_replay.py -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
python3 -m pytest tests/ --cov=src --cov-report=html
```

**Current test status**: 29 tests passing ✅

---

## 📖 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/api/models` | List available models |
| POST | `/api/analyze` | Start analysis job |
| GET | `/api/analysis/{id}` | Get analysis status |
| GET | `/api/recommendation/{id}` | Get recommendation |

### Example: Start Analysis

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "data_source": "csv",
    "file_path": "data/prompts.json",
    "limit": 100,
    "current_model": "gpt-4o"
  }'
```

---

## 📈 Sample Output

```
============================================================
  Track 4: Cost-Quality Optimization Demo
============================================================

✓ Loaded 100 prompts
✓ Generated 500 evaluation results (100 prompts × 5 models)

💰 COST IMPACT
   Current monthly cost:     $0.84
   Recommended monthly cost: $0.03
   Cost reduction:           96.0%

📈 QUALITY IMPACT
   Current quality:     94.5%
   Recommended quality: 83.9%

🎯 SEGMENT RECOMMENDATIONS
   Simple: gpt-4o → gemini-2.5-flash (96% savings)
   Medium: gpt-4o → gemini-2.5-flash (96% savings)
   Complex: gpt-4o → gemini-2.5-flash (96% savings)

🏆 PARETO FRONTIER
   Optimal models: ['gpt-4o', 'claude-haiku-4-5-20251001', 'gemini-2.5-flash']

✓ Report saved to: demo_report.html
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"

Install missing dependencies:
```bash
pip install -r requirements.txt
```

### "Connection test failed"

1. Check your `PORTKEY_API_KEY` in `.env`
2. Verify Portkey Model Catalog is set up
3. Run `python3 scripts/test_portkey.py` to diagnose

### "Rate limit exceeded"

The system has built-in rate limiting. If you hit provider limits:
- Reduce `--prompts` count
- Add delay between batches in `src/replay/rate_limiter.py`

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Portkey](https://portkey.ai/) - Unified LLM Gateway
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM Evaluation
- [BERTScore](https://github.com/Tiiiger/bert_score) - Semantic Similarity
