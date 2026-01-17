"""Repository pattern for database access."""

from typing import Optional
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import Analysis, Evaluation, Recommendation, get_db
from src.models.evaluation import EvaluationResult
from src.models.recommendation import Recommendation as RecommendationModel


class AnalysisRepository:
    """Repository for Analysis CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        analysis_id: str,
        data_source: str,
        current_model: str,
        models: list[str],
        limit: int,
        file_path: Optional[str] = None,
        include_bertscore: bool = True,
        include_deepeval: bool = False,
    ) -> Analysis:
        """Create a new analysis record."""
        analysis = Analysis(
            id=analysis_id,
            status="pending",
            data_source=data_source,
            file_path=file_path,
            current_model=current_model,
            models_compared=models,
            prompt_limit=limit,
            include_bertscore=include_bertscore,
            include_deepeval=include_deepeval,
        )
        self.session.add(analysis)
        await self.session.flush()
        return analysis
    
    async def get(self, analysis_id: str) -> Optional[Analysis]:
        """Get analysis by ID."""
        result = await self.session.execute(
            select(Analysis).where(Analysis.id == analysis_id)
        )
        return result.scalar_one_or_none()
    
    async def update_status(
        self,
        analysis_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update analysis status."""
        values = {"status": status}
        if status == "completed":
            values["completed_at"] = datetime.utcnow()
        if error_message:
            values["error_message"] = error_message
        
        await self.session.execute(
            update(Analysis)
            .where(Analysis.id == analysis_id)
            .values(**values)
        )
    
    async def update_counts(
        self,
        analysis_id: str,
        total_prompts: int,
        total_replays: int,
    ) -> None:
        """Update prompt and replay counts."""
        await self.session.execute(
            update(Analysis)
            .where(Analysis.id == analysis_id)
            .values(
                total_prompts=total_prompts,
                total_replays=total_replays,
            )
        )
    
    async def list_recent(self, limit: int = 10) -> list[Analysis]:
        """List recent analyses."""
        result = await self.session.execute(
            select(Analysis)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class EvaluationRepository:
    """Repository for Evaluation CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_batch(
        self,
        analysis_id: str,
        evaluations: list[EvaluationResult],
    ) -> int:
        """Create multiple evaluation records."""
        records = []
        for ev in evaluations:
            records.append(Evaluation(
                analysis_id=analysis_id,
                prompt_id=ev.prompt_id,
                model_id=ev.model_id,
                completion="",  # Don't store full completion to save space
                finish_reason="",
                input_tokens=ev.cost.input_tokens,
                output_tokens=ev.cost.output_tokens,
                total_tokens=ev.cost.total_tokens,
                cost_usd=ev.cost.total_cost_usd,
                composite_score=ev.composite_score,
                metrics={k: {"score": v.score, "passed": v.passed} for k, v in ev.metrics.items()},
                latency_ms=ev.latency_ms,
                success=True,
                refused=ev.refused,
            ))
        
        self.session.add_all(records)
        await self.session.flush()
        return len(records)
    
    async def get_by_analysis(self, analysis_id: str) -> list[Evaluation]:
        """Get all evaluations for an analysis."""
        result = await self.session.execute(
            select(Evaluation).where(Evaluation.analysis_id == analysis_id)
        )
        return list(result.scalars().all())
    
    async def get_by_model(self, analysis_id: str, model_id: str) -> list[Evaluation]:
        """Get evaluations for a specific model."""
        result = await self.session.execute(
            select(Evaluation)
            .where(Evaluation.analysis_id == analysis_id)
            .where(Evaluation.model_id == model_id)
        )
        return list(result.scalars().all())


class RecommendationRepository:
    """Repository for Recommendation CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        analysis_id: str,
        recommendation: RecommendationModel,
    ) -> Recommendation:
        """Create recommendation record."""
        rec = Recommendation(
            analysis_id=analysis_id,
            executive_summary=recommendation.executive_summary,
            current_monthly_cost=recommendation.current_monthly_cost,
            recommended_monthly_cost=recommendation.recommended_monthly_cost,
            cost_reduction_percent=recommendation.total_cost_reduction_percent,
            cost_savings_monthly=recommendation.total_cost_savings_monthly,
            current_avg_quality=recommendation.current_avg_quality.value,
            recommended_avg_quality=recommendation.recommended_avg_quality.value,
            quality_impact_percent=recommendation.total_quality_impact_percent,
            segments=[s.model_dump() for s in recommendation.segments],
            pareto_models=recommendation.pareto_frontier_models,
            dominated_models=recommendation.dominated_models,
            portkey_routing_config=recommendation.portkey_routing_config,
        )
        self.session.add(rec)
        await self.session.flush()
        return rec
    
    async def get_by_analysis(self, analysis_id: str) -> Optional[Recommendation]:
        """Get recommendation for an analysis."""
        result = await self.session.execute(
            select(Recommendation).where(Recommendation.analysis_id == analysis_id)
        )
        return result.scalar_one_or_none()
