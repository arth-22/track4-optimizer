"""Pareto frontier analysis for multi-objective optimization."""

import numpy as np
import structlog

from src.models.evaluation import ModelAggregateResult

logger = structlog.get_logger()


class ParetoAnalyzer:
    """
    Multi-objective optimization for cost-quality trade-offs.
    
    Finds Pareto-optimal models (non-dominated solutions) where
    no other model is better on all dimensions.
    """

    def find_pareto_frontier(
        self,
        models: list[ModelAggregateResult],
    ) -> list[ModelAggregateResult]:
        """
        Find Pareto-optimal models.
        
        A model is Pareto-optimal if no other model is:
        - Cheaper AND higher quality
        
        Args:
            models: List of model aggregate results
            
        Returns:
            List of Pareto-optimal models
        """
        if not models:
            return []

        # Extract objectives: minimize cost, maximize quality
        # Convert to minimization: use negative quality
        points = np.array([
            [m.mean_cost_per_request, -m.mean_quality_score]
            for m in models
        ])

        # Find non-dominated indices
        pareto_indices = self._find_pareto_indices(points)

        pareto_models = [models[i] for i in pareto_indices]

        logger.info(
            "Found Pareto frontier",
            total_models=len(models),
            pareto_optimal=len(pareto_models),
            dominated=len(models) - len(pareto_models),
        )

        return pareto_models

    def _find_pareto_indices(self, points: np.ndarray) -> list[int]:
        """
        Find indices of Pareto-optimal points.
        
        Uses efficient O(n²) algorithm for small n.
        """
        n = len(points)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i (j is better on all objectives)
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    is_dominated[i] = True
                    break

        return [i for i in range(n) if not is_dominated[i]]

    def find_pareto_3d(
        self,
        models: list[ModelAggregateResult],
    ) -> list[ModelAggregateResult]:
        """
        Find Pareto frontier with 3 objectives: cost, quality, latency.
        """
        if not models:
            return []

        # Minimize cost, minimize latency, maximize quality
        points = np.array([
            [m.mean_cost_per_request, m.mean_latency_ms, -m.mean_quality_score]
            for m in models
        ])

        pareto_indices = self._find_pareto_indices(points)
        return [models[i] for i in pareto_indices]

    def get_dominated_models(
        self,
        models: list[ModelAggregateResult],
        pareto_frontier: list[ModelAggregateResult],
    ) -> list[ModelAggregateResult]:
        """Get models that are dominated (not on Pareto frontier)."""
        pareto_ids = {m.model_id for m in pareto_frontier}
        return [m for m in models if m.model_id not in pareto_ids]

    def calculate_hypervolume(
        self,
        pareto_models: list[ModelAggregateResult],
        reference_point: tuple[float, float] | None = None,
    ) -> float:
        """
        Calculate hypervolume indicator for Pareto frontier quality.
        
        Higher hypervolume = better Pareto frontier.
        
        Args:
            pareto_models: Pareto-optimal models
            reference_point: Reference point (max_cost, min_quality)
            
        Returns:
            Hypervolume value
        """
        if not pareto_models:
            return 0.0

        # Default reference point: worst observed values
        if reference_point is None:
            max_cost = max(m.mean_cost_per_request for m in pareto_models) * 1.1
            min_quality = min(m.mean_quality_score for m in pareto_models) * 0.9
            reference_point = (max_cost, min_quality)

        # Sort by cost (ascending)
        sorted_models = sorted(pareto_models, key=lambda m: m.mean_cost_per_request)

        # Calculate hypervolume using inclusion-exclusion
        hypervolume = 0.0
        prev_quality = reference_point[1]

        for model in sorted_models:
            cost = model.mean_cost_per_request
            quality = model.mean_quality_score

            # Rectangle contribution
            width = reference_point[0] - cost
            height = quality - prev_quality

            if width > 0 and height > 0:
                hypervolume += width * height

            prev_quality = max(prev_quality, quality)

        return hypervolume

    def find_knee_point(
        self,
        pareto_models: list[ModelAggregateResult],
    ) -> ModelAggregateResult | None:
        """
        Find the "knee" of the Pareto frontier.
        
        The knee is the point with maximum curvature, representing
        the best trade-off between cost and quality.
        
        Returns:
            Model at the knee point, or None if insufficient data
        """
        if len(pareto_models) < 3:
            return pareto_models[0] if pareto_models else None

        # Sort by cost
        sorted_models = sorted(pareto_models, key=lambda m: m.mean_cost_per_request)

        # Calculate curvature at each point
        max_curvature = 0
        knee_model = sorted_models[len(sorted_models) // 2]  # Default to middle

        for i in range(1, len(sorted_models) - 1):
            prev_m = sorted_models[i - 1]
            curr_m = sorted_models[i]
            next_m = sorted_models[i + 1]

            # Normalized vectors
            v1_cost = (curr_m.mean_cost_per_request - prev_m.mean_cost_per_request)
            v1_qual = (curr_m.mean_quality_score - prev_m.mean_quality_score)
            v2_cost = (next_m.mean_cost_per_request - curr_m.mean_cost_per_request)
            v2_qual = (next_m.mean_quality_score - curr_m.mean_quality_score)

            # Cross product (curvature approximation)
            cross = abs(v1_cost * v2_qual - v1_qual * v2_cost)

            if cross > max_curvature:
                max_curvature = cross
                knee_model = curr_m

        return knee_model

    def generate_pareto_plot_data(
        self,
        models: list[ModelAggregateResult],
        pareto_frontier: list[ModelAggregateResult],
    ) -> dict:
        """
        Generate data for Pareto plot visualization.
        
        Returns dict suitable for Plotly chart generation.
        """
        pareto_ids = {m.model_id for m in pareto_frontier}

        all_points = []
        for m in models:
            all_points.append({
                "model_id": m.model_id,
                "provider": m.provider,
                "cost": m.mean_cost_per_request,
                "quality": m.mean_quality_score,
                "latency": m.mean_latency_ms,
                "is_pareto": m.model_id in pareto_ids,
                "sample_size": m.sample_size,
            })

        # Sort frontier by cost for line drawing
        frontier_sorted = sorted(
            [p for p in all_points if p["is_pareto"]],
            key=lambda x: x["cost"]
        )

        return {
            "all_points": all_points,
            "frontier": frontier_sorted,
            "dominated_count": len(models) - len(pareto_frontier),
        }
