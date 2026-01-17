#!/usr/bin/env python3
"""
Quick test script to verify Portkey connection.

Usage:
    python scripts/test_portkey.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL


async def test_single_model(client: AsyncOpenAI, model_slug: str) -> bool:
    """Test a single model."""
    try:
        print(f"  Testing {model_slug}...", end=" ")
        response = await client.chat.completions.create(
            model=model_slug,
            messages=[{"role": "user", "content": "Say 'hello' in one word"}],
            max_tokens=10,
        )
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or "No content"
            print(f"✅ Response: {content[:30]}...")
            return True
        else:
            print(f"⚠️ Empty response")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}...")
        return False


async def main():
    """Test Portkey connection with all configured models."""
    # Load API key from .env
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("PORTKEY_API_KEY")
    
    if not api_key:
        print("❌ PORTKEY_API_KEY not found in environment")
        print("   Set it in .env file or environment")
        return

    print("=" * 60)
    print("Portkey Connection Test")
    print("=" * 60)
    print(f"Gateway: {PORTKEY_GATEWAY_URL}")
    print(f"API Key: {api_key[:10]}...")
    print()

    client = AsyncOpenAI(
        base_url=PORTKEY_GATEWAY_URL,
        api_key=api_key,
    )

    models = [
        "@openai/gpt-4o",
        "@openai/gpt-4o-mini",
        "@anthropic/claude-sonnet-4-5",
        "@anthropic/claude-haiku-4-5-20251001",
        "@vertex/gemini-2.5-flash",
    ]

    print("Testing Models:")
    print("-" * 40)
    
    results = {}
    for model in models:
        results[model] = await test_single_model(client, model)
        await asyncio.sleep(0.5)  # Small delay between requests

    await client.close()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    print(f"Passed: {passed}/{len(models)}")
    
    if passed == len(models):
        print("\n✅ All models working! Ready for hackathon.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n⚠️  Failed models: {failed}")
        print("Check if these models are available in your Portkey Model Catalog.")


if __name__ == "__main__":
    asyncio.run(main())
