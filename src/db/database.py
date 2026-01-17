"""Database setup and session management using SQLAlchemy."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

from src.config import get_settings

Base = declarative_base()


# ============================================
# Database Models
# ============================================

class Analysis(Base):
    """Represents an analysis run."""
    
    __tablename__ = "analyses"
    
    id = Column(String(36), primary_key=True)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Configuration
    data_source = Column(String(20))  # portkey, csv, json
    file_path = Column(String(500), nullable=True)
    current_model = Column(String(100))
    models_compared = Column(JSON)  # List of model IDs
    
    # Settings
    prompt_limit = Column(Integer)
    include_bertscore = Column(Boolean, default=True)
    include_deepeval = Column(Boolean, default=False)
    
    # Results summary
    total_prompts = Column(Integer, nullable=True)
    total_replays = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="analysis")
    recommendation = relationship("Recommendation", back_populates="analysis", uselist=False)


class Evaluation(Base):
    """Stores individual evaluation results."""
    
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id"))
    prompt_id = Column(String(100))
    model_id = Column(String(100))
    
    # Completion
    completion = Column(Text)
    finish_reason = Column(String(50))
    
    # Tokens & Cost
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost_usd = Column(Float)
    
    # Quality
    composite_score = Column(Float)
    metrics = Column(JSON)  # Dict of metric scores
    
    # Performance
    latency_ms = Column(Float)
    
    # Status
    success = Column(Boolean, default=True)
    refused = Column(Boolean, default=False)
    error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="evaluations")


class Recommendation(Base):
    """Stores analysis recommendations."""
    
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("analyses.id"), unique=True)
    
    # Summary
    executive_summary = Column(Text)
    
    # Costs
    current_monthly_cost = Column(Float)
    recommended_monthly_cost = Column(Float)
    cost_reduction_percent = Column(Float)
    cost_savings_monthly = Column(Float)
    
    # Quality
    current_avg_quality = Column(Float)
    recommended_avg_quality = Column(Float)
    quality_impact_percent = Column(Float)
    
    # Recommendations
    segments = Column(JSON)  # List of segment recommendations
    pareto_models = Column(JSON)  # Pareto frontier models
    dominated_models = Column(JSON)  # Dominated models
    
    # Portkey Config
    portkey_routing_config = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="recommendation")


# ============================================
# Database Engine & Session
# ============================================

_engine = None
_session_factory = None


async def init_db(database_url: str | None = None) -> None:
    """Initialize database engine and create tables."""
    global _engine, _session_factory
    
    if database_url is None:
        settings = get_settings()
        database_url = settings.database_url
    
    # Create data directory if using SQLite
    if "sqlite" in database_url:
        db_path = database_url.replace("sqlite+aiosqlite:///", "")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    
    _engine = create_async_engine(
        database_url,
        echo=False,
        future=True,
    )
    
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_engine():
    """Get the database engine."""
    global _engine
    if _engine is None:
        await init_db()
    return _engine


async def get_session_factory():
    """Get the session factory."""
    global _session_factory
    if _session_factory is None:
        await init_db()
    return _session_factory


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session as async context manager."""
    factory = await get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Alias for cleaner imports
DatabaseSession = get_db
