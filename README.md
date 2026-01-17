# Track 4: Cost-Quality Optimization via Historical Replay

A production-grade system for analyzing LLM cost-quality trade-offs through historical prompt replay.

## Features

- **Historical Replay**: Replay prompts through alternative models via Portkey
- **Multi-dimensional Evaluation**: Quality (DeepEval, BERTScore), cost, latency
- **Statistical Analysis**: Confidence intervals, significance testing
- **Pareto Optimization**: Find optimal cost-quality trade-offs
- **Actionable Recommendations**: Quantified savings with quality impact

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your Portkey API key

# Run the API
uvicorn src.main:app --reload

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/
├── adapters/      # Data ingestion (Portkey, CSV)
├── replay/        # Model replay engine
├── evaluation/    # Quality & cost evaluation
├── analysis/      # Statistics, Pareto, recommendations
├── api/           # FastAPI routes
└── models/        # Pydantic schemas
```

## Environment Variables

```
PORTKEY_API_KEY=your_portkey_api_key
OPENAI_API_KEY=your_openai_api_key (optional, for DeepEval judge)
```

## API Endpoints

- `POST /analyze` - Run full cost-quality analysis
- `GET /models` - List available models with pricing
- `GET /recommendations/{analysis_id}` - Get optimization recommendations
- `GET /visualization/{analysis_id}` - Get Pareto chart

## License

MIT
