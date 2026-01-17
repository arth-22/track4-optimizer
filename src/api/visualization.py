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

    def generate_premium_report(
        self,
        recommendation: Recommendation,
        all_models: list[ModelAggregateResult],
        pareto_models: list[ModelAggregateResult],
        analysis_date: str = "",
        prompts_analyzed: int = 0,
    ) -> str:
        """
        Generate visually stunning, presentation-ready HTML report.
        
        Features:
        - Dark glassmorphism theme
        - Modern typography (Inter)
        - Gradient accent colors
        - Animated transitions
        - Comprehensive data visualization
        """
        from datetime import datetime
        
        if not analysis_date:
            analysis_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Calculate key metrics
        best_quality_model = max(all_models, key=lambda m: m.mean_quality_score)
        cheapest_model = min(all_models, key=lambda m: m.mean_cost_per_request)
        pareto_model_names = [m.model_id for m in pareto_models]
        
        # Generate premium-styled Plotly charts
        pareto_chart = self._generate_premium_pareto_chart(all_models, pareto_models)
        quality_chart = self._generate_premium_quality_chart(all_models)
        cost_chart = self._generate_premium_cost_chart(all_models)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cost-Quality Optimization Report | Portkey AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0b0f;
            --bg-secondary: #12141c;
            --bg-card: rgba(18, 20, 28, 0.8);
            --accent-primary: #7c3aed;
            --accent-secondary: #06b6d4;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --text-primary: #f0f6fc;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --gradient-primary: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
            --gradient-card: linear-gradient(145deg, rgba(124, 58, 237, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
            --shadow-glow: 0 0 40px rgba(124, 58, 237, 0.15);
        }}
        
        * {{ box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 24px;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            margin-bottom: 48px;
            animation: fadeIn 0.6s ease-out;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 12px 0;
        }}
        
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        .header .meta {{
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 8px;
        }}
        
        /* Hero Metrics */
        .hero-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 48px;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideUp 0.5s ease-out;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-glow);
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--gradient-primary);
        }}
        
        .metric-card .icon {{
            font-size: 2rem;
            margin-bottom: 16px;
        }}
        
        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }}
        
        .metric-card .label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 8px;
        }}
        
        .metric-card .description {{
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 4px;
        }}
        
        /* Section */
        .section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 32px;
            animation: fadeIn 0.6s ease-out;
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }}
        
        .section-header h2 {{
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }}
        
        .section-header .icon {{
            font-size: 1.5rem;
        }}
        
        /* Chart containers */
        .chart-container {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        
        /* Model Table */
        .model-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.9rem;
        }}
        
        .model-table thead tr {{
            background: var(--gradient-card);
        }}
        
        .model-table th {{
            padding: 16px;
            text-align: left;
            font-weight: 600;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-subtle);
        }}
        
        .model-table td {{
            padding: 16px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-secondary);
        }}
        
        .model-table tbody tr {{
            transition: background 0.2s ease;
        }}
        
        .model-table tbody tr:hover {{
            background: rgba(124, 58, 237, 0.1);
        }}
        
        .model-table .model-name {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .model-table .badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-pareto {{
            background: rgba(124, 58, 237, 0.2);
            color: var(--accent-primary);
        }}
        
        .badge-best {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-success);
        }}
        
        .badge-cheapest {{
            background: rgba(6, 182, 212, 0.2);
            color: var(--accent-secondary);
        }}
        
        /* Progress bars */
        .progress-bar {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
        }}
        
        .progress-bar-fill {{
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 10px;
            transition: width 1s ease-out;
        }}
        
        /* Recommendation Card */
        .recommendation-card {{
            background: var(--gradient-card);
            border: 1px solid rgba(124, 58, 237, 0.3);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
        }}
        
        .recommendation-card .segment {{
            font-weight: 600;
            color: var(--accent-primary);
            margin-bottom: 8px;
        }}
        
        .recommendation-card .arrow {{
            color: var(--text-muted);
            margin: 0 8px;
        }}
        
        .recommendation-card .savings {{
            color: var(--accent-success);
            font-weight: 600;
        }}
        
        /* Config block */
        .config-block {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 20px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            overflow-x: auto;
            white-space: pre;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--border-subtle);
        }}
        
        .footer a {{
            color: var(--accent-primary);
            text-decoration: none;
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8rem; }}
            .metric-card .value {{ font-size: 2rem; }}
            .container {{ padding: 24px 16px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 Cost-Quality Optimization Report</h1>
            <p class="subtitle">{recommendation.executive_summary}</p>
            <p class="meta">Generated {analysis_date} • Analyzed {prompts_analyzed} prompts across {len(all_models)} models</p>
        </div>
        
        <!-- Hero Metrics -->
        <div class="hero-metrics">
            <div class="metric-card" style="animation-delay: 0.1s">
                <div class="icon">💰</div>
                <div class="value">${recommendation.total_cost_savings_monthly:,.0f}</div>
                <div class="label">Monthly Savings</div>
                <div class="description">Projected savings with optimized routing</div>
            </div>
            <div class="metric-card" style="animation-delay: 0.2s">
                <div class="icon">📉</div>
                <div class="value">{recommendation.total_cost_reduction_percent:.1f}%</div>
                <div class="label">Cost Reduction</div>
                <div class="description">Percentage decrease in LLM costs</div>
            </div>
            <div class="metric-card" style="animation-delay: 0.3s">
                <div class="icon">⚡</div>
                <div class="value">{abs(recommendation.total_quality_impact_percent):.1f}%</div>
                <div class="label">Quality Impact</div>
                <div class="description">{"Improvement" if recommendation.total_quality_impact_percent > 0 else "Trade-off"} in response quality</div>
            </div>
            <div class="metric-card" style="animation-delay: 0.4s">
                <div class="icon">🏆</div>
                <div class="value">{len(pareto_models)}</div>
                <div class="label">Pareto Optimal</div>
                <div class="description">Models on the efficiency frontier</div>
            </div>
        </div>
        
        <!-- Pareto Chart -->
        <div class="section">
            <div class="section-header">
                <span class="icon">📊</span>
                <h2>Cost vs Quality Trade-off</h2>
            </div>
            <div class="chart-container">
                {pareto_chart}
            </div>
        </div>
        
        <!-- Model Comparison Table -->
        <div class="section">
            <div class="section-header">
                <span class="icon">🔍</span>
                <h2>Model Performance Comparison</h2>
            </div>
            <table class="model-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Provider</th>
                        <th>Quality Score</th>
                        <th>Cost per 1K</th>
                        <th>Avg Latency</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_model_table_rows(all_models, pareto_model_names, best_quality_model.model_id, cheapest_model.model_id)}
                </tbody>
            </table>
        </div>
        
        <!-- Quality Comparison -->
        <div class="section">
            <div class="section-header">
                <span class="icon">⚡</span>
                <h2>Quality Comparison</h2>
            </div>
            <div class="chart-container">
                {quality_chart}
            </div>
        </div>
        
        <!-- Cost Comparison -->
        <div class="section">
            <div class="section-header">
                <span class="icon">💵</span>
                <h2>Cost Comparison</h2>
            </div>
            <div class="chart-container">
                {cost_chart}
            </div>
        </div>
        
        <!-- Segment Recommendations -->
        <div class="section">
            <div class="section-header">
                <span class="icon">🎯</span>
                <h2>Routing Recommendations</h2>
            </div>
            {self._generate_premium_recommendations(recommendation.segments)}
        </div>
        
        <!-- Portkey Config -->
        <div class="section">
            <div class="section-header">
                <span class="icon">⚙️</span>
                <h2>Portkey Routing Configuration</h2>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                Copy this configuration to your Portkey Gateway for optimized routing:
            </p>
            <div class="config-block">{self._format_json(recommendation.portkey_routing_config)}</div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Generated by <a href="https://portkey.ai">Portkey AI</a> Cost-Quality Optimizer</p>
            <p>Track 4: AI Builders Challenge</p>
        </div>
    </div>
</body>
</html>'''
        return html

    def _generate_premium_pareto_chart(
        self,
        all_models: list[ModelAggregateResult],
        pareto_models: list[ModelAggregateResult],
    ) -> str:
        """Generate Pareto chart with premium dark theme."""
        pareto_ids = {m.model_id for m in pareto_models}
        
        fig = go.Figure()
        
        # Dominated models
        dominated = [m for m in all_models if m.model_id not in pareto_ids]
        if dominated:
            fig.add_trace(go.Scatter(
                x=[m.mean_cost_per_request * 1000 for m in dominated],
                y=[m.mean_quality_score for m in dominated],
                mode='markers',
                name='Other Models',
                marker=dict(size=14, color='#71717a', symbol='circle'),
                text=[f"<b>{m.model_id}</b><br>Cost: ${m.mean_cost_per_request*1000:.3f}/1K<br>Quality: {m.mean_quality_score:.1%}" for m in dominated],
                hoverinfo='text',
            ))
        
        # Pareto models
        pareto_sorted = sorted(pareto_models, key=lambda m: m.mean_cost_per_request)
        fig.add_trace(go.Scatter(
            x=[m.mean_cost_per_request * 1000 for m in pareto_sorted],
            y=[m.mean_quality_score for m in pareto_sorted],
            mode='lines+markers',
            name='Pareto Frontier',
            marker=dict(size=18, color='#7c3aed', symbol='star'),
            line=dict(color='#7c3aed', dash='dash', width=2),
            text=[f"<b>{m.model_id}</b><br>Cost: ${m.mean_cost_per_request*1000:.3f}/1K<br>Quality: {m.mean_quality_score:.1%}" for m in pareto_sorted],
            hoverinfo='text',
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f6fc', family='Inter'),
            xaxis=dict(
                title='Cost per 1,000 Requests ($)',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
            ),
            yaxis=dict(
                title='Quality Score',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickformat='.0%',
            ),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a1a1aa'),
            ),
            margin=dict(l=60, r=30, t=30, b=60),
            hovermode='closest',
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _generate_premium_quality_chart(self, models: list[ModelAggregateResult]) -> str:
        """Generate quality bar chart with gradient colors."""
        sorted_models = sorted(models, key=lambda m: m.mean_quality_score, reverse=True)
        
        # Create gradient colors
        colors = ['#7c3aed' if i == 0 else '#06b6d4' if i == 1 else '#3b82f6' for i in range(len(sorted_models))]
        
        fig = go.Figure(go.Bar(
            x=[m.model_id for m in sorted_models],
            y=[m.mean_quality_score for m in sorted_models],
            marker_color=colors,
            text=[f"{m.mean_quality_score:.1%}" for m in sorted_models],
            textposition='outside',
            textfont=dict(color='#f0f6fc'),
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f6fc', family='Inter'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='.0%',
                range=[0, 1],
            ),
            margin=dict(l=60, r=30, t=30, b=60),
            showlegend=False,
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _generate_premium_cost_chart(self, models: list[ModelAggregateResult]) -> str:
        """Generate cost bar chart with gradient colors."""
        sorted_models = sorted(models, key=lambda m: m.mean_cost_per_request)
        
        colors = ['#10b981' if i == 0 else '#06b6d4' if i == 1 else '#3b82f6' for i in range(len(sorted_models))]
        
        fig = go.Figure(go.Bar(
            x=[m.model_id for m in sorted_models],
            y=[m.mean_cost_per_request * 1000 for m in sorted_models],
            marker_color=colors,
            text=[f"${m.mean_cost_per_request*1000:.3f}" for m in sorted_models],
            textposition='outside',
            textfont=dict(color='#f0f6fc'),
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f6fc', family='Inter'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                title='Cost per 1K Requests ($)',
            ),
            margin=dict(l=60, r=30, t=30, b=60),
            showlegend=False,
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _generate_model_table_rows(
        self,
        models: list[ModelAggregateResult],
        pareto_ids: list[str],
        best_quality_id: str,
        cheapest_id: str,
    ) -> str:
        """Generate HTML table rows for models."""
        rows = []
        sorted_models = sorted(models, key=lambda m: m.mean_quality_score, reverse=True)
        
        for m in sorted_models:
            badges = []
            if m.model_id in pareto_ids:
                badges.append('<span class="badge badge-pareto">⭐ Pareto</span>')
            if m.model_id == best_quality_id:
                badges.append('<span class="badge badge-best">🏆 Best Quality</span>')
            if m.model_id == cheapest_id:
                badges.append('<span class="badge badge-cheapest">💰 Cheapest</span>')
            
            badge_html = ' '.join(badges) if badges else '<span style="color: var(--text-muted)">—</span>'
            
            rows.append(f'''
                <tr>
                    <td class="model-name">{m.model_id}</td>
                    <td>{m.provider}</td>
                    <td>
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span>{m.mean_quality_score:.1%}</span>
                            <div class="progress-bar" style="width: 80px;">
                                <div class="progress-bar-fill" style="width: {m.mean_quality_score*100}%;"></div>
                            </div>
                        </div>
                    </td>
                    <td>${m.mean_cost_per_request * 1000:.3f}</td>
                    <td>{m.mean_latency_ms:.0f}ms</td>
                    <td>{badge_html}</td>
                </tr>
            ''')
        
        return '\n'.join(rows)

    def _generate_premium_recommendations(self, segments: list[SegmentRecommendation]) -> str:
        """Generate recommendation cards."""
        cards = []
        
        for seg in segments:
            verdict_color = {
                "strong_recommendation": "var(--accent-success)",
                "recommendation": "#10b981",
                "consider": "var(--accent-warning)",
                "no_change": "var(--text-muted)",
                "not_recommended": "var(--accent-danger)",
            }.get(seg.verdict.value, "var(--text-muted)")
            
            change_text = "No change recommended" if seg.current_model == seg.recommended_model else f'{seg.current_model} → {seg.recommended_model}'
            
            cards.append(f'''
                <div class="recommendation-card">
                    <div class="segment">{seg.segment_name.upper()} • {seg.volume_percent:.0f}% of traffic</div>
                    <div style="font-size: 1.1rem; margin-bottom: 8px;">
                        {change_text}
                    </div>
                    <div>
                        <span class="savings" style="color: {verdict_color}">{seg.cost_reduction_percent:+.1f}% cost</span>
                        <span style="color: var(--text-muted); margin: 0 8px;">•</span>
                        <span style="color: {'var(--accent-success)' if seg.quality_impact_percent >= 0 else 'var(--accent-warning)'}">
                            {seg.quality_impact_percent:+.1f}% quality
                        </span>
                    </div>
                </div>
            ''')
        
        return '\n'.join(cards)
