# Track 4: Cost-Quality Optimization - Working Documentation

> **Developer Reference** | Comprehensive end-to-end documentation with module details and improvement areas

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Module Documentation](#module-documentation)
- [Production Pipeline](#production-pipeline)
- [Areas for Improvement](#areas-for-improvement)
- [Quick Reference](#quick-reference)

---

## Overview

Track 4 is a **cost-quality optimization platform** that:

1. **Ingests** historical LLM logs from Portkey
2. **Replays** prompts through multiple models via Portkey Gateway
3. **Evaluates** quality using BERTScore, LLM-as-judge, and cost tracking
4. **Analyzes** trade-offs using Pareto frontier analysis
5. **Recommends** optimal model routing strategies

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Portkey Logs  │────▶│  Replay Engine  │────▶│   Evaluators    │
│   (Historical)  │     │  (5+ Models)    │     │ (BERTScore/LLM) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌─────────────────┐             ▼
                        │   Report HTML   │◀────┌─────────────────┐
                        │   + JSON Config │     │  Pareto + Rec   │
                        └─────────────────┘     │     Engine      │
                                                └─────────────────┘
```

---

## Architecture

### Directory Structure

```
src/
├── adapters/           # Data ingestion layer
│   ├── portkey_adapter.py    # Portkey Log Export API
│   ├── csv_adapter.py        # CSV/JSON file import
│   ├── validation.py         # Data validation + sampling
│   └── base.py               # Abstract base adapter
│
├── replay/             # Model replay layer
│   ├── engine.py             # Core replay engine (Portkey Gateway)
│   ├── model_registry.py     # Model configs + pricing
│   └── rate_limiter.py       # Provider-specific rate limits
│
├── evaluation/         # Quality evaluation layer
│   ├── composite.py          # Aggregates all evaluators
│   ├── bertscore_evaluator.py    # Semantic similarity
│   ├── deepeval_evaluator.py     # LLM-as-judge
│   ├── cost_tracker.py           # Economic metrics
│   └── self_consistency.py       # Multi-sample agreement
│
├── analysis/           # Analysis + recommendation layer
│   ├── pareto.py             # Pareto frontier analysis
│   ├── recommendation.py     # Segment recommendations
│   └── statistics.py         # Statistical tests
│
├── api/                # REST API + visualization
│   ├── routes.py             # FastAPI endpoints
│   ├── visualization.py      # Plotly chart generation
│   └── schemas.py            # Request/response models
│
└── models/             # Data models (Pydantic)
    ├── canonical.py          # CanonicalPrompt, Message
    ├── evaluation.py         # EvaluationResult, ReplayResult
    └── recommendation.py     # Recommendation, SegmentRecommendation
```

---

## Data Flow

### 1. Data Ingestion

```python
# PortkeyLogAdapter (src/adapters/portkey_adapter.py)
async def fetch_prompts(start_date, end_date, limit):
    # 1. Try direct /logs API (405 expected)
    # 2. Fall back to Export API:
    #    a. create() -> get export_id
    #    b. start(export_id) <- CRITICAL: must call!
    #    c. poll status until 'success'
    #    d. download(export_id) -> get signed_url
    #    e. fetch signed_url -> NDJSON
    #    f. parse NDJSON -> list[dict]
    # 3. Normalize to CanonicalPrompt
```

**Key Classes:**
| Class | Purpose | Lines |
|-------|---------|-------|
| `PortkeyLogAdapter` | Fetch logs via Portkey Export API | 620 |
| `CSVAdapter` | Load from JSON/CSV files | 80 |
| `DataValidator` | Validate, dedupe, flag outliers | 200 |
| `SamplingStrategy` | Uniform, stratified, cost-focused sampling | 200 |

### 2. Replay Engine

```python
# ReplayEngine (src/replay/engine.py)
async def replay_batch(prompts, models, temperature):
    for prompt in prompts:
        for model in models:
            # Build Portkey Model Catalog slug: @openai/gpt-4o
            model_slug = f"@{model.provider_slug}/{model.model_id}"
            
            # Call via AsyncOpenAI + Portkey Gateway
            response = await client.chat.completions.create(
                model=model_slug,
                messages=prompt.to_openai_messages(),
                extra_headers={"x-portkey-provider": model.provider_slug}
            )
            
            yield ReplayResult(completion=..., latency_ms=..., cost=...)
```

**Key Components:**
| Component | Purpose |
|-----------|---------|
| `ReplayEngine` | Orchestrates API calls via Portkey Gateway |
| `ModelRegistry` | Contains 5 models with pricing: gpt-4o, gpt-4o-mini, claude-sonnet-4-5, claude-haiku-4-5, gemini-2.5-flash |
| `RateLimiter` | Provider-specific rate limits (OpenAI: 500/min, Anthropic: 300/min) |

### 3. Evaluation Pipeline

```python
# CompositeEvaluator (src/evaluation/composite.py)
async def evaluate(prompt, replay, reference):
    metrics = {}
    
    # BERTScore: semantic similarity (local, no API)
    if enable_bertscore:
        metrics.update(await bertscore.evaluate(...))  # P, R, F1
    
    # LLM Judge: via Portkey @openai/gpt-4o-mini
    if enable_deepeval:
        metrics.update(await llm_judge.evaluate(...))  # 0.0-1.0 score
    
    # Cost tracking
    cost = cost_tracker.calculate_cost(model_id, input_tokens, output_tokens)
    
    # Weighted composite score
    composite = weighted_average(metrics, weights)
    
    return EvaluationResult(metrics=metrics, composite_score=composite, cost=cost)
```

**Evaluators:**
| Evaluator | Method | API Calls | Produces |
|-----------|--------|-----------|----------|
| `BertScoreEvaluator` | DeBERTa embeddings | None (local) | P, R, F1 |
| `SimpleLLMJudge` | GPT-4o-mini via Portkey | 1 per eval | 0.0-1.0 + reason |
| `DeepEvalEvaluator` | DeepEval library | Via SDK | Answer relevancy |
| `CostTracker` | Token-based pricing | None | USD cost |
| `SelfConsistencyEvaluator` | Multi-sample agreement | N samples | agreement_rate |

### 4. Analysis Engine

```python
# ParetoAnalyzer (src/analysis/pareto.py)
def find_pareto_frontier(models):
    # Points: (cost, -quality) <- minimize both
    # Find non-dominated solutions
    # A model is Pareto-optimal if no other is:
    #   - Cheaper AND higher quality
    return pareto_optimal_models

# RecommendationEngine (src/analysis/recommendation.py)  
def generate_recommendation(prompts, evaluations, current_model):
    # 1. Aggregate results by model
    # 2. Segment by complexity (simple/medium/complex)
    # 3. For each segment, find best cost/quality trade-off
    # 4. Generate verdict: STRONG_REC, MODERATE_REC, NO_CHANGE, MAINTAIN
    # 5. Build Portkey routing config
    return Recommendation(
        segments=[...],
        portkey_routing_config={...}
    )
```

### 5. Output Generation

```python
# VisualizationGenerator (src/api/visualization.py)
def generate_full_report(recommendation, all_models, pareto_models):
    # 1. Pareto frontier scatter plot
    # 2. Cost breakdown stacked bars
    # 3. Savings waterfall chart
    # 4. Quality comparison bars
    # 5. Segment recommendation table
    return html_report
```

---

## Production Pipeline

### Running the Demo

```bash
# Full production demo (real Portkey data)
python3 scripts/production_demo.py --source portkey --limit 50 -y

# Quick mode (3 models, skip BERTScore)
python3 scripts/production_demo.py --source portkey --limit 20 --quick -y

# From CSV export
python3 scripts/production_demo.py --source csv --file data/logs.json --limit 100
```

### Pipeline Stages

```
1. FETCH LOGS        → PortkeyLogAdapter.fetch_prompts()
   └─ Export API: create → start → poll → download (signed_url) → NDJSON

2. VALIDATE DATA     → DataValidator.validate_batch()
   └─ Completeness check, deduplication, outlier detection

3. REPLAY PROMPTS    → ReplayEngine.replay_batch()
   └─ 5 models × N prompts = 5N API calls via Portkey

4. EVALUATE          → CompositeEvaluator.evaluate_batch()
   └─ BERTScore (local) + LLM Judge (5N API calls)

5. ANALYZE           → RecommendationEngine.generate_recommendation()
   └─ Pareto frontier + segmentation + verdict

6. GENERATE REPORT   → VisualizationGenerator.generate_full_report()
   └─ production_report.html + portkey_routing_config.json
```

---

## Module Documentation

### Adapters Module

#### PortkeyLogAdapter

**File:** `src/adapters/portkey_adapter.py` (620 lines)

**Purpose:** Fetch historical logs from Portkey Log Export API.

**Key Methods:**
```python
async def fetch_prompts(start_date, end_date, limit) -> list[CanonicalPrompt]
async def _create_export(start_date, end_date) -> dict  # SDK: create()
async def _start_export(export_id) -> None              # SDK: start() ← CRITICAL
async def _get_export_status(export_id) -> dict         # SDK: retrieve()
async def _download_export_data(export_id) -> list      # SDK: download() + signed_url
def _parse_ndjson(content) -> list[dict]                # Newline-delimited JSON
def _normalize_to_canonical(log) -> CanonicalPrompt
def _parse_timestamp(timestamp_str) -> datetime         # Multi-format parser
```

**Important Notes:**
- Must call `start()` after `create()` or export never processes
- Export response is NDJSON (newline-delimited JSON), not JSON array
- Timestamps come in multiple formats (ISO, JS Date strings)
- Message content may be list format (Anthropic style) or string

#### DataValidator

**File:** `src/adapters/validation.py` (lines 52-204)

**Purpose:** Validate and clean ingested prompts.

**Features:**
- Completeness checking (id, messages, completion)
- Format validation (non-empty content)
- Deduplication (SHA256 hash of prompt+completion)
- Outlier detection (token count > 2 std deviations)

#### SamplingStrategy

**File:** `src/adapters/validation.py` (lines 207-406)

**Strategies:**
| Method | Description |
|--------|-------------|
| `uniform_random()` | Simple random sampling |
| `stratified_by_complexity()` | Proportional by simple/medium/complex |
| `stratified_by_task_type()` | Proportional by QA/code/summarization/etc |
| `cost_focused()` | Oversample high-token prompts |
| `failure_focused()` | Prioritize low-quality prompts |

---

### Replay Module

#### ReplayEngine

**File:** `src/replay/engine.py` (355 lines)

**Architecture:**
```
AsyncOpenAI SDK
       │
       ▼
┌─────────────────────────────────────┐
│   PORTKEY_GATEWAY_URL               │
│   https://api.portkey.ai/v1         │
├─────────────────────────────────────┤
│   Headers:                          │
│   - x-portkey-api-key               │
│   - x-portkey-provider: openai      │
├─────────────────────────────────────┤
│   Model: @openai/gpt-4o             │
│          @anthropic/claude-sonnet-4-5│
│          @vertex/gemini-2.5-flash   │
└─────────────────────────────────────┘
```

**Key Features:**
- Uses `AsyncOpenAI` SDK with Portkey Gateway URL
- Model Catalog format: `@provider/model_id`
- Automatic retry with exponential backoff
- Refusal detection
- Rate limiting per provider

#### ModelRegistry

**File:** `src/replay/model_registry.py` (186 lines)

**Registered Models:**
| Model | Provider | Input/1M | Output/1M | Max Context |
|-------|----------|----------|-----------|-------------|
| `gpt-4o` | OpenAI | $2.50 | $10.00 | 128K |
| `gpt-4o-mini` | OpenAI | $0.15 | $0.60 | 128K |
| `claude-sonnet-4-5` | Anthropic | $3.00 | $15.00 | 200K |
| `claude-haiku-4-5` | Anthropic | $0.80 | $4.00 | 200K |
| `gemini-2.5-flash` | Vertex | $0.10 | $0.40 | 1M |

---

### Evaluation Module

#### CompositeEvaluator

**File:** `src/evaluation/composite.py` (212 lines)

**Default Weights:**
```python
DEFAULT_WEIGHTS = {
    "bertscore_f1": 0.3,
    "deepeval_answer_relevancy": 0.4,
    "llm_judge_score": 0.3,
}
```

**Flow:**
1. Run each enabled evaluator
2. Collect metrics dict
3. Calculate cost via CostTracker
4. Compute weighted composite score
5. Return EvaluationResult

#### BertScoreEvaluator

**File:** `src/evaluation/bertscore_evaluator.py` (183 lines)

**Configuration:**
```python
BertScoreEvaluator(
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en",  # Required for rescale_with_baseline
    rescale_with_baseline=True
)
```

**Output Metrics:** `bertscore_precision`, `bertscore_recall`, `bertscore_f1`

#### SimpleLLMJudge

**File:** `src/evaluation/deepeval_evaluator.py` (lines 167-290)

**Judge Prompt:**
```
Rate the response on a scale of 0.0 to 1.0:
1. Relevance: Does it answer the question?
2. Accuracy: Is the information correct?
3. Completeness: Does it cover key points?
4. Coherence: Is it well-organized?

Respond with ONLY: {"score": 0.0-1.0, "reason": "..."}
```

**Model:** `@openai/gpt-4o-mini` (via Portkey)

---

### Analysis Module

#### ParetoAnalyzer

**File:** `src/analysis/pareto.py` (233 lines)

**Algorithm:**
```python
# For each model, check if any other model dominates it
# A model is dominated if another is: 
#   - Equal or better on ALL objectives
#   - Strictly better on AT LEAST ONE objective

for model_i in models:
    for model_j in models:
        if all(j <= i) and any(j < i):  # j dominates i
            mark i as dominated
```

**Additional Methods:**
- `find_pareto_3d()` - Include latency as 3rd objective
- `calculate_hypervolume()` - Quality metric for frontier
- `find_knee_point()` - Best cost/quality trade-off

#### RecommendationEngine

**File:** `src/analysis/recommendation.py` (549 lines)

**Segmentation:**
```python
# Segment prompts by complexity
segments = {
    "simple": prompts where total_tokens < 500,
    "medium": prompts where 500 <= total_tokens <= 3000,
    "complex": prompts where total_tokens > 3000
}

# For each segment, recommend best model
for segment in segments:
    current_cost = sum(costs for current_model)
    best_model = min(models, key=lambda m: cost / quality)
    savings = current_cost - best_model_cost
```

**Verdicts:**
| Verdict | Criteria |
|---------|----------|
| `STRONG_RECOMMENDATION` | >20% savings, <5% quality drop |
| `MODERATE_RECOMMENDATION` | 10-20% savings |
| `NO_CHANGE` | <5% savings |
| `MAINTAIN_CURRENT` | Recommended = current |

---

## Areas for Improvement

### ✅ Completed (100% Hackathon Requirements)

#### 1. ~~Direct Logs API~~ → ✅ Export API Works
**Status:** Confirmed via web search - Portkey only supports Export API, not direct logs endpoint.

#### 2. ~~BERTScore Language Detection~~ → ✅ Implemented
**Location:** `src/evaluation/bertscore_evaluator.py`
- Auto-detects 11 languages via `langdetect`
- Caches scorers per language
- Hash-based caching for repeated evaluations

#### 3. ~~Async Close Methods~~ → ✅ Fixed
**Location:** `production_demo.py`
- Uses `await adapter.aclose()` and `await engine.close()`

#### 4. ~~Evaluation Parallelization~~ → ✅ Implemented
**Location:** `src/evaluation/composite.py`
- `asyncio.gather()` with `Semaphore(10)` for rate limiting
- 10x faster batch evaluation

#### 5. ~~BERTScore Caching~~ → ✅ Implemented
**Location:** `src/evaluation/bertscore_evaluator.py`
- SHA256 hash-based cache keys
- Module-level `_bertscore_cache` dict

#### 6. ~~Model Registry Pricing~~ → ✅ Implemented
**Location:** `src/replay/model_registry.py` + `data/pricing.json`
- `load_pricing_from_file()` method
- `save_pricing_to_file()` for backups

#### 7. ~~PII Scrubbing~~ → ✅ Implemented
**Location:** `src/adapters/validation.py`
- `PIIScrubber` class with 5 PII patterns (email, phone, SSN, credit card, IP)

#### 8. ~~Streaming Replay~~ → ✅ Implemented
**Location:** `src/replay/engine.py`
- `replay_single_streaming()` method using OpenAI stream API

#### 9. ~~Daemon Mode~~ → ✅ Implemented
**Location:** `scripts/production_demo.py`
- `--daemon` flag for continuous execution
- `--interval` to configure hours between runs

#### 10. ~~Multi-Language Support~~ → ✅ Implemented
**Location:** `src/models/canonical.py`
- `content_language` field added to `CanonicalPrompt`

---

### 🆕 New Production Features

#### Guardrail Evaluator
**File:** `src/evaluation/guardrails.py`

```python
from src.evaluation import GuardrailEvaluator

guard = GuardrailEvaluator()
result = await guard.evaluate_response(prompt, response)
# Returns: {passed, toxicity, pii_leakage, prompt_injection}
```

**Detects:**
- Toxicity (pattern-based)
- PII leakage
- Prompt injection attempts

#### Anomaly Detector
**File:** `src/analysis/anomaly.py`

```python
from src.analysis import AnomalyDetector

detector = AnomalyDetector()
result = await detector.detect_anomalies(evaluations)
# Returns: {anomalies, insights, recommended_actions}
```

**Features:**
- LLM-powered pattern analysis
- Detects unusual cost/quality patterns
- Provides actionable insights

#### Graceful Degradation
**File:** `src/evaluation/composite.py`

When an evaluator fails, automatically falls back to basic text similarity:

```python
# If BERTScore fails → basic word overlap F1
# If LLM Judge fails → basic word overlap F1
```

---

## Quick Reference

### Environment Variables

```bash
# Required
PORTKEY_API_KEY=your-api-key

# Optional
PORTKEY_WORKSPACE_ID=ws-xxx      # For log export
DATABASE_URL=sqlite:///data.db   # For persistence
OPENAI_API_KEY=xxx               # For DeepEval (if not using Portkey)
```

### Key Commands

```bash
# Test Portkey connection
python3 scripts/test_portkey.py

# Run production demo
python3 scripts/production_demo.py --source portkey --limit 50 -y

# Start API server
python3 -m uvicorn src.main:app --reload

# Run tests
python3 -m pytest tests/ -v
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Start analysis job |
| `/analyze/{job_id}` | GET | Get analysis status |
| `/recommendation/{id}` | GET | Get recommendation |

### Output Files

| File | Contents |
|------|----------|
| `production_report.html` | Interactive charts + tables |
| `portkey_routing_config.json` | Ready-to-import routing config |

---

## Contributing

1. All new evaluators must extend `BaseEvaluator`
2. All new adapters must extend `BaseAdapter`  
3. Add pricing to `ModelRegistry` when adding new models
4. Run `pytest` before submitting PRs
5. Keep this documentation updated

---

*Last updated: January 2026*
