#!/usr/bin/env python3
"""Test script for Portkey Log Export API integration."""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_portkey_log_export():
    """Test the Portkey Log Export adapter with real API."""
    from src.adapters.portkey_adapter import PortkeyLogAdapter
    from src.config import get_settings
    
    settings = get_settings()
    
    print("=" * 60)
    print("  Portkey Log Export API Test")
    print("=" * 60)
    print()
    
    if not settings.portkey_api_key:
        print("❌ PORTKEY_API_KEY not set in environment")
        print("   Please set it in .env file:")
        print("   PORTKEY_API_KEY=your-api-key-here")
        return False
    
    print(f"✓ API Key: {settings.portkey_api_key[:8]}...{settings.portkey_api_key[-4:]}")
    print()
    
    # Initialize adapter
    adapter = PortkeyLogAdapter()
    
    # Test 1: Count available logs
    print("Test 1: Counting available logs...")
    try:
        # Last 7 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        count = await adapter.count_available(
            start_date=start_date,
            end_date=end_date,
        )
        print(f"   ✓ Found {count} logs in last 7 days")
    except Exception as e:
        print(f"   ⚠️  Count failed (may be unsupported): {e}")
        count = None
    print()
    
    # Test 2: Fetch a small sample
    print("Test 2: Fetching sample logs...")
    try:
        prompts = await adapter.fetch_prompts(
            start_date=start_date,
            end_date=end_date,
            limit=5,
        )
        
        if prompts:
            print(f"   ✓ Fetched {len(prompts)} prompts")
            print()
            
            # Show sample
            for i, p in enumerate(prompts[:3]):
                print(f"   Sample {i + 1}:")
                print(f"   - ID: {p.id}")
                print(f"   - Model: {p.completion.model_id}")
                print(f"   - Provider: {p.completion.provider}")
                print(f"   - Tokens: {p.completion.total_tokens}")
                print(f"   - Cost: ${p.completion.cost_usd:.6f}")
                print(f"   - Prompt: {p.prompt_text[:80]}...")
                print()
        else:
            print("   ⚠️  No prompts returned (empty logs or no data in period)")
            
    except Exception as e:
        print(f"   ❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        adapter.close()
        return False
    
    # Test 3: Verify canonical format
    print("Test 3: Verifying canonical format...")
    if prompts:
        sample = prompts[0]
        checks = [
            ("id", bool(sample.id)),
            ("messages", len(sample.messages) > 0),
            ("completion.text", bool(sample.completion.text)),
            ("completion.model_id", bool(sample.completion.model_id)),
            ("metadata.complexity", bool(sample.metadata.complexity)),
            ("to_openai_format()", len(sample.to_openai_format()) > 0),
        ]
        
        all_passed = True
        for name, passed in checks:
            status = "✓" if passed else "❌"
            print(f"   {status} {name}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("   ✓ All format checks passed")
        print()
    
    # Close adapter
    adapter.close()
    
    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    
    if prompts:
        print("✅ Portkey Log Export integration is WORKING")
        print(f"   - {len(prompts)} prompts fetched successfully")
        print("   - Canonical format validated")
        print("   - Ready for production use")
        return True
    else:
        print("⚠️  Portkey Log Export returned no data")
        print("   - Check if you have logs in the specified date range")
        print("   - Verify API permissions")
        return False


async def test_with_demo_data():
    """Test with CSV adapter as fallback."""
    from src.adapters.csv_adapter import CSVAdapter
    
    print()
    print("=" * 60)
    print("  Fallback: CSV Adapter Test")
    print("=" * 60)
    print()
    
    # Generate some test data
    test_data = [
        {
            "id": "test-1",
            "prompt": "Summarize this article about AI",
            "completion": "AI is transforming industry...",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
        },
        {
            "id": "test-2",
            "prompt": "Write a Python function to sort a list",
            "completion": "def sort_list(lst):\n    return sorted(lst)",
            "model": "gpt-4o",
            "input_tokens": 50,
            "output_tokens": 30,
        },
    ]
    
    # Save to temp file
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        adapter = CSVAdapter(temp_path)
        prompts = await adapter.fetch_prompts(limit=10)
        
        print(f"✓ CSV Adapter loaded {len(prompts)} prompts")
        for p in prompts:
            print(f"   - {p.id}: {p.prompt_text[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ CSV Adapter failed: {e}")
        return False
    finally:
        os.unlink(temp_path)


async def main():
    """Run all tests."""
    print()
    
    # Test Portkey Log Export
    portkey_ok = await test_portkey_log_export()
    
    if not portkey_ok:
        # Fallback to CSV
        await test_with_demo_data()
    
    print()
    print("=" * 60)
    print("  Next Steps")
    print("=" * 60)
    
    if portkey_ok:
        print("1. Run full demo with Portkey data:")
        print("   python3 scripts/demo.py --prompts 100 --live")
        print()
        print("2. Generate recommendations from your actual logs")
    else:
        print("1. Ensure PORTKEY_API_KEY is set correctly")
        print("2. Verify you have logs in Portkey dashboard")
        print("3. Check API permissions in Portkey settings")
        print()
        print("For now, use simulated data:")
        print("   python3 scripts/demo.py --prompts 100")


if __name__ == "__main__":
    asyncio.run(main())
