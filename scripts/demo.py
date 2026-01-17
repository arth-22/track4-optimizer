#!/usr/bin/env python3
"""
Demo script for Track 4: Cost-Quality Optimization.

This script demonstrates the full pipeline:
1. Load sample data
2. Replay through multiple models
3. Evaluate quality
4. Generate recommendations
5. Display results

Usage:
    python scripts/demo.py [--prompts N] [--live]
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.csv_adapter import CSVAdapter
from src.analysis.pareto import ParetoAnalyzer
from src.analysis.recommendation import RecommendationEngine
from src.api.visualization import VisualizationGenerator
from src.evaluation.composite import CompositeEvaluator
from src.models.canonical import (
    CanonicalPrompt,
    CompletionData,
    Message,
    MessageRole,
    PromptMetadata,
    ComplexityLevel,
    TaskType,
)
from src.replay.engine import ReplayEngine
from src.replay.model_registry import ModelRegistry


def create_demo_prompts(count: int = 50) -> list[CanonicalPrompt]:
    """Generate demo prompts for demonstration."""
    prompts = []

    # Sample prompts with varying complexity
    scenarios = [
        # Simple prompts
        ("What is 2+2?", "4", ComplexityLevel.SIMPLE, TaskType.QA),
        ("Translate 'hello' to Spanish", "Hola", ComplexityLevel.SIMPLE, TaskType.TRANSLATION),
        ("What color is the sky?", "Blue", ComplexityLevel.SIMPLE, TaskType.QA),
        # Medium prompts
        ("Explain how machine learning works", "Machine learning is...", ComplexityLevel.MEDIUM, TaskType.QA),
        ("Summarize the theory of relativity", "Einstein's theory...", ComplexityLevel.MEDIUM, TaskType.SUMMARIZATION),
        ("What are the benefits of cloud computing?", "Cloud computing offers...", ComplexityLevel.MEDIUM, TaskType.QA),
        # Complex prompts
        ("Write a Python function to implement quicksort", "def quicksort(arr)...", ComplexityLevel.COMPLEX, TaskType.CODE_GENERATION),
        ("Analyze the economic impact of AI on employment", "The impact of AI...", ComplexityLevel.COMPLEX, TaskType.REASONING),
        ("Design a microservices architecture for e-commerce", "The architecture...", ComplexityLevel.COMPLEX, TaskType.REASONING),
    ]

    for i in range(count):
        scenario = scenarios[i % len(scenarios)]
        prompt_text, completion_text, complexity, task_type = scenario

        prompts.append(CanonicalPrompt(
            id=f"demo-{i:04d}",
            source="demo",
            created_at=datetime.utcnow(),
            messages=[
                Message(role=MessageRole.USER, content=prompt_text),
            ],
            completion=CompletionData(
                text=completion_text,
                model_id="gpt-4o",
                provider="openai",
                input_tokens=len(prompt_text) // 4 + 10,
                output_tokens=len(completion_text) // 4 + 20,
                total_tokens=(len(prompt_text) + len(completion_text)) // 4 + 30,
                latency_ms=500 + (i % 5) * 200,
                cost_usd=0.001 + (i % 3) * 0.0005,
                created_at=datetime.utcnow(),
            ),
            metadata=PromptMetadata(
                task_type=task_type,
                complexity=complexity,
            ),
        ))

    return prompts


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print formatted section."""
    print(f"\n>>> {text}")
    print("-" * 40)


async def run_demo(prompt_count: int = 50, live: bool = False):
    """Run the demo pipeline."""
    print_header("Track 4: Cost-Quality Optimization Demo")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Mode: {'Live API' if live else 'Simulated'}")

    # 1. Generate/Load Data
    print_section("1. Loading Data")
    prompts = create_demo_prompts(prompt_count)
    print(f"✓ Loaded {len(prompts)} prompts")

    # Show complexity distribution
    complexity_counts = {}
    for p in prompts:
        c = p.metadata.complexity.value
        complexity_counts[c] = complexity_counts.get(c, 0) + 1
    print(f"  Complexity distribution: {complexity_counts}")

    # 2. Replay (simulated for demo)
    print_section("2. Replaying through models")
    models = ModelRegistry.get_default_models()
    print(f"  Target models: {models}")

    if live:
        print("  ⚠️  Live mode not implemented in demo")
        print("  Using simulated results instead")

    # Simulate replay results
    from src.models.evaluation import ReplayResult, EvaluationResult, MetricScore, CostResult
    import random

    evaluations = []
    for prompt in prompts:
        for model_id in models:
            # Simulate quality based on model and complexity
            base_quality = {
                "gpt-4o": 0.92,
                "gpt-4o-mini": 0.85,
                "claude-sonnet-4-20250514": 0.91,
                "claude-haiku-4-20250514": 0.82,
                "gemini-2.0-flash": 0.80,
            }.get(model_id, 0.85)

            # Complex prompts benefit more from better models
            complexity_bonus = {
                ComplexityLevel.SIMPLE: 0.0,
                ComplexityLevel.MEDIUM: 0.02,
                ComplexityLevel.COMPLEX: 0.05,
            }.get(prompt.metadata.complexity, 0.0)

            if model_id in ["gpt-4o", "claude-sonnet-4-20250514"]:
                quality = base_quality + complexity_bonus + random.uniform(-0.05, 0.05)
            else:
                quality = base_quality - complexity_bonus * 0.5 + random.uniform(-0.05, 0.05)

            quality = max(0.5, min(1.0, quality))

            # Get cost from registry
            pricing = ModelRegistry.get_pricing(model_id)
            input_tokens = prompt.completion.input_tokens
            output_tokens = prompt.completion.output_tokens
            cost = pricing.calculate_cost(input_tokens, output_tokens) if pricing else 0.001

            evaluations.append(EvaluationResult(
                prompt_id=prompt.id,
                model_id=model_id,
                metrics={
                    "composite": MetricScore(
                        metric_name="composite",
                        score=quality,
                        reason="Simulated evaluation",
                        passed=quality > 0.7,
                    ),
                },
                composite_score=quality,
                cost=CostResult(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    input_cost_usd=cost * 0.3,
                    output_cost_usd=cost * 0.7,
                    total_cost_usd=cost,
                    model_id=model_id,
                ),
                latency_ms=300 + random.randint(0, 500),
            ))

    print(f"✓ Generated {len(evaluations)} evaluation results")

    # 3. Generate Recommendation
    print_section("3. Generating Recommendations")
    rec_engine = RecommendationEngine()
    recommendation = rec_engine.generate_recommendation(
        prompts=prompts,
        evaluations=evaluations,
        current_model="gpt-4o",
    )

    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"   Prompts analyzed: {recommendation.total_prompts_analyzed:,}")
    print(f"   Models compared: {len(recommendation.models_evaluated)}")

    print(f"\n💰 COST IMPACT")
    print(f"   Current monthly cost:     ${recommendation.current_monthly_cost:,.2f}")
    print(f"   Recommended monthly cost: ${recommendation.recommended_monthly_cost:,.2f}")
    print(f"   Monthly savings:          ${recommendation.total_cost_savings_monthly:,.2f}")
    print(f"   Cost reduction:           {recommendation.total_cost_reduction_percent:.1f}%")

    print(f"\n📈 QUALITY IMPACT")
    print(f"   Current quality:     {recommendation.current_avg_quality.value:.1%}")
    print(f"   Recommended quality: {recommendation.recommended_avg_quality.value:.1%}")
    print(f"   Quality change:      {recommendation.total_quality_impact_percent:+.1f}%")

    print(f"\n🎯 SEGMENT RECOMMENDATIONS")
    for seg in recommendation.segments:
        print(f"   {seg.segment_name}: {seg.current_model} → {seg.recommended_model}")
        print(f"      Savings: {seg.cost_reduction_percent:.1f}% | Quality: {seg.quality_impact_percent:+.1f}%")
        print(f"      Verdict: {seg.verdict.value.replace('_', ' ').title()}")

    print(f"\n🏆 PARETO FRONTIER")
    print(f"   Optimal models: {recommendation.pareto_frontier_models}")
    print(f"   Dominated models: {recommendation.dominated_models}")

    # 4. Generate Report
    print_section("4. Generating Report")
    model_results = rec_engine._aggregate_by_model(evaluations)
    pareto = ParetoAnalyzer()
    pareto_models = pareto.find_pareto_frontier(model_results)

    viz = VisualizationGenerator()
    report_html = viz.generate_full_report(recommendation, model_results, pareto_models)

    # Save report
    report_path = Path(__file__).parent.parent / "demo_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)

    print(f"✓ Report saved to: {report_path}")

    # 5. Portkey Routing Config
    print_section("5. Portkey Routing Config")
    print(json.dumps(recommendation.portkey_routing_config, indent=2))

    print_header("Demo Complete!")
    print(f"Open {report_path} to view the interactive report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track 4 Demo")
    parser.add_argument("--prompts", type=int, default=50, help="Number of demo prompts")
    parser.add_argument("--live", action="store_true", help="Use live API (requires keys)")
    args = parser.parse_args()

    asyncio.run(run_demo(prompt_count=args.prompts, live=args.live))
