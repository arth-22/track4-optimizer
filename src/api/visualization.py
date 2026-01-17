"""Plotly visualization generator for analysis results."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import structlog

from src.models.evaluation import ModelAggregateResult
from src.models.recommendation import Recommendation, SegmentRecommendation

logger = structlog.get_logger()


class VisualizationGenerator:
    """
    Generates interactive Plotly visualizations for analysis results.
    
    Creates:
    - Pareto frontier charts (cost vs quality)
    - Cost breakdown by segment
    - Quality comparison radar charts
    - Savings waterfall charts
    """

    def __init__(self, template: str = "plotly_white"):
        self.template = template

    def generate_pareto_chart(
        self,
        all_models: list[ModelAggregateResult],
        pareto_models: list[ModelAggregateResult],
        title: str = "Cost vs Quality Trade-off",
    ) -> str:
        """
        Generate interactive Pareto frontier chart.
        
        Returns HTML string with embedded Plotly chart.
        """
        pareto_ids = {m.model_id for m in pareto_models}

        fig = go.Figure()

        # Non-Pareto models (dominated)
        dominated = [m for m in all_models if m.model_id not in pareto_ids]
        if dominated:
            fig.add_trace(go.Scatter(
                x=[m.mean_cost_per_request * 1000 for m in dominated],  # Cost per 1000 requests
                y=[m.mean_quality_score for m in dominated],
                mode='markers',
                name='Dominated Models',
                marker=dict(
                    size=12,
                    color='lightgray',
                    symbol='circle',
                ),
                text=[f"{m.model_id}<br>Provider: {m.provider}" for m in dominated],
                hovertemplate="<b>%{text}</b><br>Cost: $%{x:.4f}/1K<br>Quality: %{y:.2%}<extra></extra>",
            ))

        # Pareto frontier models
        pareto_sorted = sorted(pareto_models, key=lambda m: m.mean_cost_per_request)
        fig.add_trace(go.Scatter(
            x=[m.mean_cost_per_request * 1000 for m in pareto_sorted],
            y=[m.mean_quality_score for m in pareto_sorted],
            mode='lines+markers',
            name='Pareto Frontier',
            marker=dict(
                size=16,
                color='blue',
                symbol='star',
            ),
            line=dict(
                color='blue',
                dash='dash',
                width=2,
            ),
            text=[f"{m.model_id}<br>Provider: {m.provider}" for m in pareto_sorted],
            hovertemplate="<b>%{text}</b><br>Cost: $%{x:.4f}/1K<br>Quality: %{y:.2%}<extra></extra>",
        ))

        # Add model labels
        for m in pareto_sorted:
            fig.add_annotation(
                x=m.mean_cost_per_request * 1000,
                y=m.mean_quality_score,
                text=m.model_id.split('-')[0],  # Short name
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Cost per 1,000 Requests ($)",
            yaxis_title="Quality Score",
            template=self.template,
            hovermode='closest',
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
            ),
        )

        fig.update_yaxes(tickformat='.0%')

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def generate_cost_breakdown_chart(
        self,
        segments: list[SegmentRecommendation],
        title: str = "Cost Breakdown by Segment",
    ) -> str:
        """Generate stacked bar chart showing cost breakdown."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Current Cost", "Recommended Cost"),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )

        # Current costs pie
        fig.add_trace(go.Pie(
            labels=[s.segment_name for s in segments],
            values=[s.current_cost_monthly for s in segments],
            name="Current",
            textinfo='label+percent',
            hovertemplate="%{label}<br>$%{value:,.2f}/month<extra></extra>",
        ), row=1, col=1)

        # Recommended costs pie
        fig.add_trace(go.Pie(
            labels=[s.segment_name for s in segments],
            values=[s.recommended_cost_monthly for s in segments],
            name="Recommended",
            textinfo='label+percent',
            hovertemplate="%{label}<br>$%{value:,.2f}/month<extra></extra>",
        ), row=1, col=2)

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            template=self.template,
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def generate_savings_waterfall(
        self,
        recommendation: Recommendation,
        title: str = "Monthly Savings Breakdown",
    ) -> str:
        """Generate waterfall chart showing savings breakdown."""
        segments = recommendation.segments

        # Build waterfall data
        categories = ["Current Cost"]
        values = [recommendation.current_monthly_cost]
        measures = ["absolute"]

        for seg in segments:
            categories.append(f"{seg.segment_name} Savings")
            values.append(-seg.cost_savings_monthly)
            measures.append("relative")

        categories.append("Recommended Cost")
        values.append(recommendation.recommended_monthly_cost)
        measures.append("total")

        fig = go.Figure(go.Waterfall(
            name="Cost",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            textposition="outside",
            text=[f"${abs(v):,.0f}" for v in values],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}},
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            yaxis_title="Monthly Cost ($)",
            template=self.template,
            showlegend=False,
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def generate_quality_comparison(
        self,
        models: list[ModelAggregateResult],
        title: str = "Model Quality Comparison",
    ) -> str:
        """Generate grouped bar chart comparing model quality."""
        fig = go.Figure()

        # Sort by quality score
        sorted_models = sorted(models, key=lambda m: m.mean_quality_score, reverse=True)

        fig.add_trace(go.Bar(
            x=[m.model_id for m in sorted_models],
            y=[m.mean_quality_score for m in sorted_models],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[m.quality_ci_upper - m.mean_quality_score for m in sorted_models],
                arrayminus=[m.mean_quality_score - m.quality_ci_lower for m in sorted_models],
            ),
            marker_color=[
                'blue' if m.mean_quality_score == sorted_models[0].mean_quality_score
                else 'lightblue'
                for m in sorted_models
            ],
            text=[f"{m.mean_quality_score:.1%}" for m in sorted_models],
            textposition='outside',
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Model",
            yaxis_title="Quality Score",
            template=self.template,
            yaxis=dict(tickformat='.0%', range=[0, 1]),
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def generate_latency_distribution(
        self,
        models: list[ModelAggregateResult],
        title: str = "Latency Distribution by Model",
    ) -> str:
        """Generate box plot showing latency distribution."""
        fig = go.Figure()

        for m in models:
            fig.add_trace(go.Box(
                name=m.model_id,
                q1=[m.mean_latency_ms * 0.8],  # Approximation
                median=[m.p50_latency_ms],
                q3=[m.p95_latency_ms],
                lowerfence=[m.mean_latency_ms * 0.5],
                upperfence=[m.p99_latency_ms],
                mean=[m.mean_latency_ms],
                boxpoints=False,
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            yaxis_title="Latency (ms)",
            template=self.template,
            showlegend=False,
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def generate_segment_recommendation_table(
        self,
        segments: list[SegmentRecommendation],
    ) -> str:
        """Generate HTML table with segment recommendations."""
        headers = [
            "Segment", "Volume", "Current Model", "Recommended",
            "Cost Savings", "Quality Impact", "Verdict"
        ]

        rows = []
        for seg in segments:
            verdict_color = {
                "strong_recommendation": "green",
                "recommendation": "lightgreen",
                "consider": "yellow",
                "no_change": "gray",
                "not_recommended": "red",
            }.get(seg.verdict.value, "gray")

            rows.append([
                seg.segment_name,
                f"{seg.volume_percent:.1f}%",
                seg.current_model,
                seg.recommended_model,
                f"-{seg.cost_reduction_percent:.1f}%",
                f"{seg.quality_impact_percent:+.1f}%",
                f'<span style="color:{verdict_color}">●</span> {seg.verdict.value.replace("_", " ").title()}',
            ])

        # Build HTML table
        html = '<table style="width:100%; border-collapse:collapse; font-family:sans-serif;">'
        html += '<thead><tr style="background:#f0f0f0;">'
        for h in headers:
            html += f'<th style="padding:10px; text-align:left; border-bottom:2px solid #ddd;">{h}</th>'
        html += '</tr></thead><tbody>'

        for row in rows:
            html += '<tr>'
            for cell in row:
                html += f'<td style="padding:10px; border-bottom:1px solid #eee;">{cell}</td>'
            html += '</tr>'

        html += '</tbody></table>'
        return html

    def generate_full_report(
        self,
        recommendation: Recommendation,
        all_models: list[ModelAggregateResult],
        pareto_models: list[ModelAggregateResult],
    ) -> str:
        """Generate full HTML report with all visualizations."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cost-Quality Optimization Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin-right: 30px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .chart {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>🎯 Cost-Quality Optimization Report</h1>
            
            <div class="summary">
                <p>{recommendation.executive_summary}</p>
                <div class="metric">
                    <div class="metric-value">${recommendation.total_cost_savings_monthly:,.0f}</div>
                    <div class="metric-label">Monthly Savings</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{recommendation.total_cost_reduction_percent:.1f}%</div>
                    <div class="metric-label">Cost Reduction</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{abs(recommendation.total_quality_impact_percent):.1f}%</div>
                    <div class="metric-label">Quality Impact</div>
                </div>
            </div>

            <h2>📊 Cost vs Quality Trade-off</h2>
            <div class="chart">{self.generate_pareto_chart(all_models, pareto_models)}</div>

            <h2>💰 Savings Breakdown</h2>
            <div class="chart">{self.generate_savings_waterfall(recommendation)}</div>

            <h2>📋 Segment Recommendations</h2>
            {self.generate_segment_recommendation_table(recommendation.segments)}

            <h2>⚡ Quality Comparison</h2>
            <div class="chart">{self.generate_quality_comparison(all_models)}</div>

            <h2>🔧 Portkey Routing Config</h2>
            <pre style="background:#f5f5f5; padding:15px; border-radius:5px; overflow-x:auto;">
{self._format_json(recommendation.portkey_routing_config)}
            </pre>
        </body>
        </html>
        """
        return html

    def _format_json(self, obj: dict) -> str:
        """Format dict as indented JSON string."""
        import json
        return json.dumps(obj, indent=2)
