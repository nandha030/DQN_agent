#!/usr/bin/env python3
"""
Dheera GUI Features Demo (CLI version)
Demonstrates hot-swappable LLM backends without requiring GUI installation
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.llm_router import LLMRouter, LLMConfig, PROVIDER_PRESETS


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")


def demo_1_basic_usage():
    """Demo 1: Basic LLM Router Usage"""
    print_header("DEMO 1: Basic LLM Router Usage")

    print("\n📦 Creating LLM Router...")
    router = LLMRouter()

    print("✅ Router created")
    print(f"   Active provider: {router.active_provider or 'None'}")

    print("\n📝 Adding Ollama phi3:mini provider...")
    success = router.add_provider("local_phi3", PROVIDER_PRESETS["ollama_phi3"])

    if success:
        print("✅ Provider added successfully!")
        print(f"   Active provider: {router.active_provider}")
    else:
        print("❌ Failed to add provider")
        return

    print("\n💬 Sending test message...")
    response = router.generate("Say hello in one sentence.")

    print(f"\n📤 Query: 'Say hello in one sentence.'")
    print(f"📥 Response: {response.text}")
    print(f"\n📊 Metadata:")
    print(f"   Provider: {response.provider}")
    print(f"   Model: {response.model}")
    print(f"   Latency: {response.latency_ms:.0f}ms")
    print(f"   Tokens: {response.tokens_used}")
    print(f"   Status: {response.finish_reason}")

    if response.error:
        print(f"   Error: {response.error}")


def demo_2_multiple_providers():
    """Demo 2: Adding Multiple Providers"""
    print_header("DEMO 2: Multiple Providers & Hot-Swapping")

    router = LLMRouter()

    print("\n📦 Adding multiple providers...")

    # Add phi3
    print("\n1️⃣  Adding: phi3:mini (local)")
    router.add_provider("phi3", PROVIDER_PRESETS["ollama_phi3"])
    print("   ✅ Added")

    # Add gemma
    print("\n2️⃣  Adding: gemma:2b (local)")
    gemma_config = LLMConfig(
        provider="ollama",
        model="gemma:2b",
        base_url="http://localhost:11434",
        timeout=15,
        max_tokens=256,
    )
    router.add_provider("gemma", gemma_config)
    print("   ✅ Added")

    print("\n📋 Listing all providers:")
    providers = router.list_providers()
    for p in providers:
        active_icon = "✅" if p["active"] else "⭕"
        print(f"   {active_icon} {p['name']:<15} {p['provider']:<10} {p['model']}")

    print(f"\n🔄 Current active provider: {router.active_provider}")


def demo_3_hot_swapping():
    """Demo 3: Hot-Swapping Providers"""
    print_header("DEMO 3: Hot-Swapping Between Providers")

    router = LLMRouter()

    # Add providers
    router.add_provider("phi3", PROVIDER_PRESETS["ollama_phi3"])
    router.add_provider("gemma", LLMConfig(
        provider="ollama",
        model="gemma:2b",
        base_url="http://localhost:11434",
        timeout=15,
        max_tokens=128,
    ))

    test_query = "What is 2+2? Answer in one word."

    # Test with phi3
    print("\n1️⃣  Using: phi3:mini")
    print(f"   Active: {router.active_provider}")

    start = time.time()
    response1 = router.generate(test_query)
    time1 = (time.time() - start) * 1000

    print(f"   Response: {response1.text[:100]}")
    print(f"   Latency: {time1:.0f}ms")

    # Hot-swap to gemma
    print("\n🔄 Hot-swapping to gemma:2b...")
    router.switch_provider("gemma")
    print(f"   ✅ Switched! Active: {router.active_provider}")

    # Test with gemma
    print("\n2️⃣  Using: gemma:2b (after hot-swap)")

    start = time.time()
    response2 = router.generate(test_query)
    time2 = (time.time() - start) * 1000

    print(f"   Response: {response2.text[:100]}")
    print(f"   Latency: {time2:.0f}ms")

    # Compare
    print("\n📊 Comparison:")
    print(f"   phi3:mini  → {time1:.0f}ms")
    print(f"   gemma:2b   → {time2:.0f}ms")
    print(f"   Difference → {abs(time1 - time2):.0f}ms")


def demo_4_statistics():
    """Demo 4: Provider Statistics"""
    print_header("DEMO 4: Provider Statistics & Monitoring")

    router = LLMRouter()
    router.add_provider("local", PROVIDER_PRESETS["ollama_phi3"])

    print("\n📊 Initial stats:")
    stats = router.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Active provider: {stats['active_provider']}")

    print("\n💬 Sending 3 test queries...")
    queries = [
        "Count to 3",
        "Say hello",
        "What is AI?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: '{query}'")
        response = router.generate(query)
        print(f"   → Latency: {response.latency_ms:.0f}ms, Tokens: {response.tokens_used}")

    print("\n📊 Updated stats:")
    stats = router.get_stats()
    print(f"   Total requests: {stats['total_requests']}")

    for provider in stats['providers']:
        print(f"\n   Provider: {provider['name']}")
        pstats = provider['stats']
        print(f"     Requests: {pstats['requests']}")
        print(f"     Total tokens: {pstats['total_tokens']}")
        print(f"     Avg latency: {pstats['total_latency_ms'] / max(pstats['requests'], 1):.0f}ms")
        print(f"     Errors: {pstats['errors']}")


def demo_5_provider_testing():
    """Demo 5: Testing Providers Before Use"""
    print_header("DEMO 5: Testing Providers")

    router = LLMRouter()
    router.add_provider("phi3", PROVIDER_PRESETS["ollama_phi3"])
    router.add_provider("gemma", LLMConfig(
        provider="ollama",
        model="gemma:2b",
        base_url="http://localhost:11434",
        timeout=15,
    ))

    print("\n🔬 Testing all providers...\n")

    providers = router.list_providers()
    for provider in providers:
        name = provider['name']
        print(f"Testing: {name} ({provider['model']})...")

        result = router.test_provider(name)

        if result['success']:
            print(f"  ✅ Working")
            print(f"     Latency: {result['latency_ms']:.0f}ms")
            print(f"     Response: {result['response'][:50]}")
        else:
            print(f"  ❌ Failed")
            print(f"     Error: {result['error'][:80]}")
        print()


def demo_6_api_simulation():
    """Demo 6: Simulating GUI API Calls"""
    print_header("DEMO 6: Simulating GUI API Interactions")

    router = LLMRouter()

    print("\n🌐 Simulating API endpoints...\n")

    # Simulate: GET /api/llm/providers
    print("📡 GET /api/llm/providers")
    router.add_provider("local", PROVIDER_PRESETS["ollama_phi3"])
    providers = router.list_providers()
    print(f"   Response: {len(providers)} provider(s)")
    for p in providers:
        print(f"     - {p['name']} ({p['model']})")

    # Simulate: POST /api/llm/provider
    print("\n📡 POST /api/llm/provider (add new)")
    new_config = LLMConfig(
        provider="ollama",
        model="qwen2:1.5b",
        base_url="http://localhost:11434",
    )
    success = router.add_provider("qwen", new_config)
    print(f"   Response: {{'success': {success}, 'name': 'qwen'}}")

    # Simulate: POST /api/llm/switch
    print("\n📡 POST /api/llm/switch")
    success = router.switch_provider("qwen")
    print(f"   Response: {{'success': {success}, 'active': '{router.active_provider}'}}")

    # Simulate: POST /api/chat
    print("\n📡 POST /api/chat")
    response = router.generate("Hello!")
    print(f"   Response: {{")
    print(f"     'text': '{response.text[:60]}...',")
    print(f"     'latency_ms': {response.latency_ms:.0f},")
    print(f"     'tokens': {response.tokens_used}")
    print(f"   }}")

    # Simulate: GET /api/stats
    print("\n📡 GET /api/stats")
    stats = router.get_stats()
    print(f"   Response: {{")
    print(f"     'total_requests': {stats['total_requests']},")
    print(f"     'active_provider': '{stats['active_provider']}'")
    print(f"   }}")


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🎨 Dheera GUI Features Demo" + " " * 25 + "║")
    print("║" + " " * 10 + "Hot-Swappable LLM Backends (CLI Version)" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\nThis demo shows all GUI features without requiring Streamlit/FastAPI.")
    print("The full web GUI provides a visual interface for these capabilities.")

    demos = [
        ("Basic Usage", demo_1_basic_usage),
        ("Multiple Providers", demo_2_multiple_providers),
        ("Hot-Swapping", demo_3_hot_swapping),
        ("Statistics", demo_4_statistics),
        ("Provider Testing", demo_5_provider_testing),
        ("API Simulation", demo_6_api_simulation),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()

            if i < len(demos):
                print("\n" + "─" * 70)
                input("Press Enter to continue to next demo...")
                print("\n" * 2)

        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in demo {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  ✅ ALL DEMOS COMPLETE!")
    print("=" * 70)

    print("\n📚 What you saw:")
    print("  ✅ Hot-swappable LLM providers")
    print("  ✅ Real-time provider switching")
    print("  ✅ Statistics tracking")
    print("  ✅ Provider testing")
    print("  ✅ API simulation")

    print("\n🚀 To use the full web GUI:")
    print("  1. Install: pip install streamlit plotly fastapi uvicorn")
    print("  2. Start backend: python3 api/server.py")
    print("  3. Start GUI: streamlit run gui_streamlit.py")
    print("  4. Open: http://localhost:8501")

    print("\n🎉 All features demonstrated!")
    print()


if __name__ == "__main__":
    main()
