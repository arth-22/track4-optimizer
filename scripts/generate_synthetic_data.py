#!/usr/bin/env python3
"""
Generate synthetic data for testing and demos.

Creates realistic prompt-completion pairs with:
- Varied complexity levels
- Different task types
- Realistic token counts and costs
- Multiple models represented

Usage:
    python scripts/generate_synthetic_data.py [--count N] [--output FILE]
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# Sample prompts by category
PROMPT_TEMPLATES = {
    "simple_qa": [
        "What is {topic}?",
        "Define {term}.",
        "Who invented {thing}?",
        "When was {event}?",
        "Where is {place} located?",
    ],
    "medium_qa": [
        "Explain how {concept} works.",
        "What are the main differences between {a} and {b}?",
        "Describe the process of {process}.",
        "What are the benefits and drawbacks of {topic}?",
        "How does {technology} impact {field}?",
    ],
    "complex_reasoning": [
        "Analyze the implications of {topic} on {field} over the next decade.",
        "Compare and contrast {a}, {b}, and {c} in terms of {criteria}.",
        "Design a solution for {problem} considering {constraints}.",
        "Evaluate the effectiveness of {approach} for {goal}.",
        "Propose a strategy for {objective} given {circumstances}.",
    ],
    "summarization": [
        "Summarize the key points about {topic}.",
        "Provide a brief overview of {subject}.",
        "What are the main takeaways from {source}?",
    ],
    "code_generation": [
        "Write a Python function to {task}.",
        "Implement a JavaScript class for {purpose}.",
        "Create a SQL query to {objective}.",
        "Write unit tests for {component}.",
        "Refactor this code to improve {aspect}.",
    ],
    "translation": [
        "Translate '{phrase}' to {language}.",
        "How do you say '{expression}' in {language}?",
    ],
    "creative": [
        "Write a short story about {theme}.",
        "Compose a poem inspired by {subject}.",
        "Create a marketing tagline for {product}.",
    ],
}

FILL_INS = {
    "topic": ["machine learning", "blockchain", "quantum computing", "climate change", "renewable energy"],
    "term": ["API", "microservices", "containerization", "DevOps", "agile methodology"],
    "thing": ["the telephone", "the internet", "the automobile", "penicillin", "electricity"],
    "event": ["World War II", "the moon landing", "the invention of the printing press"],
    "place": ["the Great Wall of China", "Machu Picchu", "the Sahara Desert"],
    "concept": ["neural networks", "supply chain optimization", "cloud computing", "encryption"],
    "a": ["React", "Python", "cloud storage", "SQL databases", "monolithic architecture"],
    "b": ["Vue", "JavaScript", "on-premise storage", "NoSQL databases", "microservices"],
    "c": ["Angular", "TypeScript", "hybrid storage", "graph databases", "serverless"],
    "technology": ["artificial intelligence", "5G networks", "IoT", "blockchain"],
    "field": ["healthcare", "finance", "education", "manufacturing", "entertainment"],
    "process": ["photosynthesis", "machine learning training", "software deployment"],
    "problem": ["data privacy", "scalability", "latency", "security vulnerabilities"],
    "constraints": ["limited budget", "time constraints", "legacy systems", "compliance requirements"],
    "approach": ["agile development", "test-driven development", "pair programming"],
    "goal": ["increasing user engagement", "reducing costs", "improving reliability"],
    "objective": ["database migration", "API integration", "performance optimization"],
    "circumstances": ["remote team", "tight deadline", "legacy codebase"],
    "criteria": ["performance", "cost", "ease of use", "scalability"],
    "task": ["sort a list", "validate email addresses", "parse JSON data", "calculate fibonacci"],
    "purpose": ["user authentication", "data validation", "API client", "state management"],
    "component": ["login form", "shopping cart", "search functionality"],
    "aspect": ["readability", "performance", "maintainability"],
    "phrase": ["Hello, world", "Good morning", "Thank you", "Happy birthday"],
    "expression": ["congratulations", "I love you", "excuse me"],
    "language": ["Spanish", "French", "German", "Japanese", "Mandarin"],
    "theme": ["artificial intelligence", "space exploration", "time travel"],
    "subject": ["nature", "technology", "human connection"],
    "product": ["productivity app", "fitness tracker", "sustainable clothing"],
    "source": ["the research paper", "the quarterly report", "the meeting notes"],
}

MODELS = [
    ("gpt-4o", "openai", 2.50, 10.00),
    ("gpt-4o-mini", "openai", 0.15, 0.60),
    ("claude-sonnet-4-20250514", "anthropic", 3.00, 15.00),
    ("claude-haiku-4-20250514", "anthropic", 0.80, 4.00),
    ("gemini-2.0-flash", "google", 0.10, 0.40),
]


def fill_template(template: str) -> str:
    """Fill in a template with random values."""
    result = template
    for key, values in FILL_INS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_completion(prompt: str, complexity: str) -> str:
    """Generate a realistic completion based on prompt complexity."""
    base_length = {
        "simple": random.randint(20, 50),
        "medium": random.randint(100, 300),
        "complex": random.randint(300, 800),
    }.get(complexity, 100)

    # Simulate a response
    words = [
        "This", "is", "a", "simulated", "response", "that", "demonstrates",
        "the", "expected", "output", "format", "for", prompt[:20], ".",
        "The", "answer", "involves", "several", "key", "points", ".",
        "First", ",", "we", "need", "to", "consider", "the", "context", ".",
        "Additionally", ",", "there", "are", "multiple", "factors", ".",
        "In", "conclusion", ",", "the", "analysis", "shows", "that", "...",
    ]

    word_count = base_length // 5
    return " ".join(random.choices(words, k=word_count))


def generate_record(index: int, base_date: datetime) -> dict:
    """Generate a single synthetic record."""
    # Choose category and complexity
    category_weights = {
        "simple_qa": 0.25,
        "medium_qa": 0.25,
        "complex_reasoning": 0.15,
        "summarization": 0.10,
        "code_generation": 0.15,
        "translation": 0.05,
        "creative": 0.05,
    }

    category = random.choices(
        list(category_weights.keys()),
        weights=list(category_weights.values()),
    )[0]

    complexity = {
        "simple_qa": "simple",
        "medium_qa": "medium",
        "complex_reasoning": "complex",
        "summarization": "medium",
        "code_generation": "complex",
        "translation": "simple",
        "creative": "medium",
    }[category]

    task_type = {
        "simple_qa": "qa",
        "medium_qa": "qa",
        "complex_reasoning": "reasoning",
        "summarization": "summarization",
        "code_generation": "code_generation",
        "translation": "translation",
        "creative": "creative_writing",
    }[category]

    # Generate prompt and completion
    template = random.choice(PROMPT_TEMPLATES[category])
    prompt = fill_template(template)
    completion = generate_completion(prompt, complexity)

    # Choose model
    model_id, provider, input_price, output_price = random.choice(MODELS)

    # Calculate tokens (approximate: 4 chars per token)
    input_tokens = len(prompt) // 4 + random.randint(5, 20)
    output_tokens = len(completion) // 4 + random.randint(10, 50)

    # Calculate cost
    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

    # Random timestamp within range
    timestamp = base_date - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    # Latency based on model and complexity
    base_latency = {
        "gpt-4o": 800,
        "gpt-4o-mini": 300,
        "claude-sonnet-4-20250514": 700,
        "claude-haiku-4-20250514": 250,
        "gemini-2.0-flash": 200,
    }.get(model_id, 500)

    complexity_multiplier = {"simple": 0.8, "medium": 1.0, "complex": 1.5}[complexity]
    latency = base_latency * complexity_multiplier * random.uniform(0.8, 1.2)

    return {
        "id": f"syn-{index:06d}",
        "created_at": timestamp.isoformat(),
        "prompt": prompt,
        "completion": completion,
        "model": model_id,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": round(cost, 6),
        "latency_ms": round(latency, 1),
        "task_type": task_type,
        "complexity": complexity,
        "finish_reason": "stop" if random.random() > 0.02 else "length",
    }


def generate_dataset(count: int, output_path: Path):
    """Generate full synthetic dataset."""
    print(f"Generating {count} synthetic records...")

    base_date = datetime.utcnow()
    records = [generate_record(i, base_date) for i in range(count)]

    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(records, indent=2, fp=f)

    print(f"Saved to: {output_path}")

    # Print statistics
    complexity_counts = {}
    model_counts = {}
    task_counts = {}

    for r in records:
        complexity_counts[r["complexity"]] = complexity_counts.get(r["complexity"], 0) + 1
        model_counts[r["model"]] = model_counts.get(r["model"], 0) + 1
        task_counts[r["task_type"]] = task_counts.get(r["task_type"], 0) + 1

    print("\nDataset Statistics:")
    print(f"  Total records: {len(records)}")
    print(f"  Complexity: {complexity_counts}")
    print(f"  Models: {model_counts}")
    print(f"  Task types: {task_counts}")

    total_cost = sum(r["cost"] for r in records)
    print(f"  Total simulated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--count", type=int, default=1000, help="Number of records")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_prompts.json",
        help="Output file path",
    )
    args = parser.parse_args()

    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_dataset(args.count, output_path)
