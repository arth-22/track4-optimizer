"""FastAPI routes for the optimization API."""

import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse
import structlog

from src.adapters import CSVAdapter, PortkeyLogAdapter
from src.analysis import ParetoAnalyzer, RecommendationEngine, StatisticalAnalyzer
from src.api.visualization import VisualizationGenerator
from src.config import get_settings
from src.evaluation import CompositeEvaluator
from src.models.recommendation import AnalysisRequest, AnalysisStatus, Recommendation
from src.replay import ModelRegistry, ReplayEngine

logger = structlog.get_logger()
router = APIRouter()

# In-memory storage for analyses (use DB in production)
_analyses: dict[str, dict[str, Any]] = {}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/models")
async def list_models():
    """List available models with pricing."""
    models = ModelRegistry.list_models()
    return {
        "models": [
            {
                "model_id": m.model_id,
                "provider": m.provider.value,
                "pricing": {
                    "input_per_million": m.pricing.input_per_million,
                    "output_per_million": m.pricing.output_per_million,
                },
                "max_context": m.max_context,
                "capabilities": m.capabilities,
            }
            for m in models
        ],
        "default_models": ModelRegistry.get_default_models(),
    }


@router.post("/analyze", response_model=AnalysisStatus)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a new cost-quality analysis.
    
    Returns an analysis ID that can be used to check status and get results.
    """
    analysis_id = str(uuid.uuid4())

    # Initialize analysis state
    _analyses[analysis_id] = {
        "status": "pending",
        "request": request,
        "prompts": [],
        "replays": [],
        "evaluations": [],
        "recommendation": None,
        "started_at": datetime.utcnow(),
    }

    # Run analysis in background
    background_tasks.add_task(run_analysis, analysis_id, request)

    return AnalysisStatus(
        analysis_id=analysis_id,
        status="pending",
        progress_percent=0.0,
        current_step="Initializing",
        started_at=datetime.utcnow(),
    )


async def run_analysis(analysis_id: str, request: AnalysisRequest):
    """Background task to run the full analysis pipeline."""
    try:
        state = _analyses[analysis_id]
        state["status"] = "ingesting"

        # 1. Ingest data
        logger.info("Starting data ingestion", analysis_id=analysis_id)
        if request.source_type == "portkey":
            adapter = PortkeyLogAdapter()
            prompts = await adapter.fetch_prompts(
                start_date=request.start_date or datetime.utcnow() - timedelta(days=30),
                end_date=request.end_date or datetime.utcnow(),
                limit=request.sample_size,
            )
        elif request.source_type in ("csv", "json"):
            if not request.source_path:
                raise ValueError("source_path required for file-based sources")
            adapter = CSVAdapter(request.source_path)
            prompts = await adapter.fetch_prompts(limit=request.sample_size)
        else:
            raise ValueError(f"Unknown source type: {request.source_type}")

        state["prompts"] = prompts
        state["prompts_ingested"] = len(prompts)
        state["status"] = "replaying"

        # 2. Replay through models
        logger.info("Starting replay", analysis_id=analysis_id, prompt_count=len(prompts))
        engine = ReplayEngine()
        replays = []
        
        async for replay in engine.replay_batch(
            prompts=prompts,
            models=request.models_to_test,
        ):
            replays.append(replay)
            state["replays_completed"] = len(replays)

        state["replays"] = replays
        state["status"] = "evaluating"

        # 3. Evaluate quality
        logger.info("Starting evaluation", analysis_id=analysis_id, replay_count=len(replays))
        evaluator = CompositeEvaluator(
            enable_bertscore=request.enable_bertscore,
            enable_deepeval=request.enable_deepeval,
        )

        # Match replays to prompts
        prompt_lookup = {p.id: p for p in prompts}
        evaluations = []
        
        for replay in replays:
            prompt = prompt_lookup.get(replay.prompt_id)
            if prompt:
                eval_result = await evaluator.evaluate(prompt, replay)
                evaluations.append(eval_result)
                state["evaluations_completed"] = len(evaluations)

        state["evaluations"] = evaluations
        state["status"] = "analyzing"

        # 4. Generate recommendation
        logger.info("Generating recommendation", analysis_id=analysis_id)
        rec_engine = RecommendationEngine()
        recommendation = rec_engine.generate_recommendation(
            prompts=prompts,
            evaluations=evaluations,
            current_model=prompts[0].completion.model_id if prompts else None,
        )

        state["recommendation"] = recommendation
        state["status"] = "complete"
        state["completed_at"] = datetime.utcnow()

        logger.info("Analysis complete", analysis_id=analysis_id)

    except Exception as e:
        logger.error("Analysis failed", analysis_id=analysis_id, error=str(e))
        _analyses[analysis_id]["status"] = "failed"
        _analyses[analysis_id]["errors"] = [str(e)]


@router.get("/analyze/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get status of a running analysis."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    state = _analyses[analysis_id]
    
    total_replays = len(state.get("prompts", [])) * 5  # Estimate: 5 models
    completed = state.get("replays_completed", 0)
    progress = (completed / total_replays * 100) if total_replays > 0 else 0

    return AnalysisStatus(
        analysis_id=analysis_id,
        status=state["status"],
        progress_percent=min(99, progress) if state["status"] != "complete" else 100,
        current_step=state["status"].title(),
        prompts_ingested=state.get("prompts_ingested", 0),
        replays_completed=state.get("replays_completed", 0),
        evaluations_completed=state.get("evaluations_completed", 0),
        started_at=state["started_at"],
        completed_at=state.get("completed_at"),
        errors=state.get("errors", []),
    )


@router.get("/analyze/{analysis_id}/recommendation")
async def get_recommendation(analysis_id: str):
    """Get the recommendation from a completed analysis."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    state = _analyses[analysis_id]
    
    if state["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Status: {state['status']}",
        )

    return state["recommendation"]


@router.get("/analyze/{analysis_id}/report", response_class=HTMLResponse)
async def get_report(analysis_id: str):
    """Get full HTML report for a completed analysis."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    state = _analyses[analysis_id]
    
    if state["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Status: {state['status']}",
        )

    # Generate visualizations
    viz = VisualizationGenerator()
    rec_engine = RecommendationEngine()
    
    # Aggregate model results
    model_results = rec_engine._aggregate_by_model(state["evaluations"])
    pareto = ParetoAnalyzer()
    pareto_models = pareto.find_pareto_frontier(model_results)

    html = viz.generate_full_report(
        recommendation=state["recommendation"],
        all_models=model_results,
        pareto_models=pareto_models,
    )

    return HTMLResponse(content=html)


@router.get("/analyze/{analysis_id}/pareto", response_class=HTMLResponse)
async def get_pareto_chart(analysis_id: str):
    """Get standalone Pareto chart for an analysis."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    state = _analyses[analysis_id]
    
    if state["status"] != "complete":
        raise HTTPException(status_code=400, detail="Analysis not complete")

    rec_engine = RecommendationEngine()
    model_results = rec_engine._aggregate_by_model(state["evaluations"])
    pareto = ParetoAnalyzer()
    pareto_models = pareto.find_pareto_frontier(model_results)

    viz = VisualizationGenerator()
    chart_html = viz.generate_pareto_chart(model_results, pareto_models)

    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head><title>Pareto Frontier</title></head>
        <body style="margin:20px;">{chart_html}</body>
        </html>
    """)


@router.get("/analyze/{analysis_id}/routing-config")
async def get_routing_config(analysis_id: str):
    """Get Portkey routing configuration from analysis."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    state = _analyses[analysis_id]
    
    if state["status"] != "complete":
        raise HTTPException(status_code=400, detail="Analysis not complete")

    recommendation: Recommendation = state["recommendation"]
    return recommendation.portkey_routing_config
