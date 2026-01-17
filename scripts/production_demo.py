#!/usr/bin/env python3
"""
Production Demo Script for Track 4: Cost-Quality Optimization.

This script uses REAL data and API calls:
1. Fetches REAL logs from Portkey Log Export API (or CSV export)
2. Replays prompts through REAL model APIs via Portkey Gateway
3. Evaluates quality using REAL BERTScore + DeepEval metrics
4. Generates recommendations based on actual performance data

Usage:
    # With Portkey logs (requires PORTKEY_API_KEY)
    python scripts/production_demo.py --source portkey --limit 50

    # With CSV/JSON export
    python scripts/production_demo.py --source csv --file data/logs.json --limit 50

    # Minimal test (10 prompts, faster)
    python scripts/production_demo.py --source portkey --limit 10 --quick
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import PortkeyLogAdapter, CSVAdapter, DataValidator, SamplingStrategy
from src.analysis.pareto import ParetoAnalyzer
from src.analysis.recommendation import RecommendationEngine
from src.api.visualization import VisualizationGenerator
from src.config import get_settings
from src.evaluation.composite import CompositeEvaluator
from src.models.evaluation import EvaluationResult
from src.replay.engine import ReplayEngine
from src.replay.model_registry import ModelRegistry

import structlog

logger = structlog.get_logger()


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """Print formatted section."""
    print(f"\n>>> {text}")
    print("-" * 50)


def print_cost_warning(num_prompts: int, num_models: int):
    """Print estimated cost warning."""
    # Rough estimates
    replay_cost = num_prompts * num_models * 0.002  # ~$0.002 per completion
    eval_cost = num_prompts * 0.01  # ~$0.01 per evaluation (DeepEval)
    total_cost = replay_cost + eval_cost
    
    print("\n⚠️  ESTIMATED API COSTS:")
    print(f"   Prompts: {num_prompts}")
    print(f"   Models: {num_models}")
    print(f"   Replay cost: ~${replay_cost:.2f}")
    print(f"   Evaluation cost: ~${eval_cost:.2f}")
    print(f"   TOTAL: ~${total_cost:.2f}")
    print()


async def fetch_real_logs(
    source: str,
    file_path: str | None,
    limit: int,
    days_back: int = 7,
) -> list:
    """Fetch REAL logs from Portkey or CSV."""
    
    if source == "portkey":
        print("📥 Fetching logs from Portkey Log Export API...")
        
        adapter = PortkeyLogAdapter()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            prompts = await adapter.fetch_prompts(
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
            await adapter.aclose()  # Proper async close
            
            if not prompts:
                print("⚠️  No logs found in Portkey for the last 7 days")
                print("   Please ensure you have made API calls through Portkey")
                return []
            
            print(f"✓ Fetched {len(prompts)} real prompts from Portkey")
            return prompts
            
        except Exception as e:
            await adapter.aclose()  # Proper async close
            logger.error("Failed to fetch Portkey logs", error=str(e))
            print(f"❌ Failed to fetch Portkey logs: {e}")
            return []
    
    elif source == "csv":
        if not file_path or not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return []
        
        print(f"📥 Loading logs from {file_path}...")
        adapter = CSVAdapter(file_path)
        prompts = await adapter.fetch_prompts(limit=limit)
        print(f"✓ Loaded {len(prompts)} prompts from file")
        return prompts
    
    else:
        print(f"❌ Unknown source: {source}")
        return []


async def replay_through_models(
    prompts: list,
    model_ids: list[str],
    quick_mode: bool = False,
) -> list:
    """Replay prompts through REAL model APIs via Portkey."""
    
    print("🔄 Replaying prompts through models via Portkey Gateway...")
    print(f"   Models: {model_ids}")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Total API calls: {len(prompts) * len(model_ids)}")
    
    if quick_mode:
        print("   Mode: Quick (limited retries)")
    
    engine = ReplayEngine()
    all_results = []
    
    # Progress tracking
    total = len(prompts) * len(model_ids)
    completed = 0
    
    try:
        async for result in engine.replay_batch(
            prompts=prompts,
            models=model_ids,
            temperature=0.7,
        ):
            all_results.append(result)
            completed += 1
            
            if completed % 10 == 0 or completed == total:
                print(f"   Progress: {completed}/{total} ({100*completed/total:.0f}%)")
    
    except Exception as e:
        logger.error("Replay failed", error=str(e))
        print(f"❌ Replay error: {e}")
    
    finally:
        await engine.close()  # Now properly awaited
    
    print(f"✓ Completed {len(all_results)} replay requests")
    
    # Count successes/failures
    successes = sum(1 for r in all_results if r.success)
    failures = len(all_results) - successes
    print(f"   Successes: {successes}, Failures: {failures}")
    
    return all_results


async def evaluate_results(
    prompts: list,
    replay_results: list,
    enable_bertscore: bool = True,
    enable_deepeval: bool = True,
) -> list[EvaluationResult]:
    """Evaluate replay results using REAL metrics."""
    
    print("📊 Evaluating quality with real metrics...")
    print(f"   BERTScore: {'Enabled' if enable_bertscore else 'Disabled'}")
    print(f"   DeepEval: {'Enabled' if enable_deepeval else 'Disabled'}")
    
    evaluator = CompositeEvaluator(
        enable_bertscore=enable_bertscore,
        enable_deepeval=enable_deepeval,
        use_simple_judge=True,  # Use simple judge for faster evaluation
    )
    
    # Create prompt lookup
    prompt_lookup = {p.id: p for p in prompts}
    
    evaluations = []
    total = len(replay_results)
    
    for i, replay in enumerate(replay_results):
        prompt = prompt_lookup.get(replay.prompt_id)
        if not prompt:
            continue
        
        try:
            # Use original completion as reference
            reference = prompt.completion.text if prompt.completion else None
            
            result = await evaluator.evaluate(
                prompt=prompt,
                replay=replay,
                reference=reference,
            )
            evaluations.append(result)
            
        except Exception as e:
            logger.warning("Evaluation failed", prompt_id=replay.prompt_id, error=str(e))
        
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"   Progress: {i+1}/{total} ({100*(i+1)/total:.0f}%)")
    
    print(f"✓ Completed {len(evaluations)} evaluations")
    
    return evaluations


async def run_production_demo(
    source: str = "portkey",
    file_path: str | None = None,
    limit: int = 50,
    quick_mode: bool = False,
    skip_confirmation: bool = False,
):
    """Run the full production pipeline with REAL data."""
    
    print_header("Track 4: Production Cost-Quality Optimization")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Source: {source}")
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    
    settings = get_settings()
    if not settings.portkey_api_key:
        print("\n❌ PORTKEY_API_KEY not set!")
        print("   Please add it to your .env file:")
        print("   PORTKEY_API_KEY=your-api-key-here")
        return
    
    # Get models
    model_ids = ModelRegistry.get_default_models()
    if quick_mode:
        model_ids = model_ids[:3]  # Use fewer models in quick mode
    
    # Cost warning
    print_cost_warning(limit, len(model_ids))
    
    if not skip_confirmation:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # === STEP 1: Fetch REAL logs ===
    print_section("1. Fetching Real Data")
    prompts = await fetch_real_logs(source, file_path, limit)
    
    if not prompts:
        print("\n❌ No prompts available. Please check your data source.")
        return
    
    # Validate data
    validator = DataValidator()
    validation = validator.validate_batch(prompts)
    prompts = validation.valid_prompts
    
    print(f"   Valid prompts: {validation.valid_count}")
    print(f"   Invalid: {validation.invalid_count}")
    print(f"   Duplicates removed: {validation.duplicates_removed}")
    
    # Show complexity distribution
    complexity_counts = {}
    for p in prompts:
        c = p.metadata.complexity.value
        complexity_counts[c] = complexity_counts.get(c, 0) + 1
    print(f"   Complexity distribution: {complexity_counts}")
    
    # === STEP 2: Replay through REAL APIs ===
    print_section("2. Replaying Through Real Model APIs")
    replay_results = await replay_through_models(
        prompts=prompts,
        model_ids=model_ids,
        quick_mode=quick_mode,
    )
    
    if not replay_results:
        print("\n❌ No replay results. Check API connectivity.")
        return
    
    # === STEP 3: Evaluate with REAL metrics ===
    print_section("3. Evaluating with Real Metrics")
    evaluations = await evaluate_results(
        prompts=prompts,
        replay_results=replay_results,
        enable_bertscore=not quick_mode,  # Skip BERTScore in quick mode
        enable_deepeval=True,
    )
    
    if not evaluations:
        print("\n❌ No evaluations completed.")
        return
    
    # === STEP 4: Generate Recommendations ===
    print_section("4. Generating Recommendations")
    
    # Get current model from prompts
    current_models = [p.completion.model_id for p in prompts if p.completion]
    most_common_model = max(set(current_models), key=current_models.count) if current_models else "gpt-4o"
    
    rec_engine = RecommendationEngine()
    recommendation = rec_engine.generate_recommendation(
        prompts=prompts,
        evaluations=evaluations,
        current_model=most_common_model,
    )
    
    # === Display Results ===
    print(f"\n{'='*70}")
    print("  📊 ANALYSIS RESULTS (REAL DATA)")
    print(f"{'='*70}")
    
    print(f"\n📊 SUMMARY")
    print(f"   Prompts analyzed: {recommendation.total_prompts_analyzed:,}")
    print(f"   Models compared: {len(recommendation.models_evaluated)}")
    print(f"   Current model: {most_common_model}")
    
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
    
    # === STEP 5: Generate Report ===
    print_section("5. Generating Report")
    
    model_results = rec_engine._aggregate_by_model(evaluations)
    pareto = ParetoAnalyzer()
    pareto_models = pareto.find_pareto_frontier(model_results)
    
    viz = VisualizationGenerator()
    report_html = viz.generate_full_report(recommendation, model_results, pareto_models)
    
    # Save report
    report_path = Path(__file__).parent.parent / "production_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    print(f"✓ Report saved to: {report_path}")
    
    # Save Portkey routing config
    config_path = Path(__file__).parent.parent / "portkey_routing_config.json"
    with open(config_path, "w") as f:
        json.dump(recommendation.portkey_routing_config, f, indent=2)
    print(f"✓ Routing config saved to: {config_path}")
    
    print_section("6. Portkey Routing Config")
    print(json.dumps(recommendation.portkey_routing_config, indent=2))
    
    print_header("Production Demo Complete!")
    print(f"\n📄 Open {report_path} to view the interactive report.")
    print(f"📋 Import {config_path} into Portkey Gateway to apply recommendations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track 4 Production Demo - Uses REAL Portkey data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch from Portkey API (requires PORTKEY_API_KEY in .env)
    python scripts/production_demo.py --source portkey --limit 50
    
    # Use exported CSV/JSON file
    python scripts/production_demo.py --source csv --file data/logs.json
    
    # Quick test (fewer models, faster)
    python scripts/production_demo.py --source portkey --limit 10 --quick
        """
    )
    
    parser.add_argument(
        "--source",
        choices=["portkey", "csv"],
        default="portkey",
        help="Data source: 'portkey' (API) or 'csv' (file)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to CSV/JSON file (required if source=csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of prompts to analyze",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer models, skip heavy evaluations",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_production_demo(
        source=args.source,
        file_path=args.file,
        limit=args.limit,
        quick_mode=args.quick,
        skip_confirmation=args.yes,
    ))
