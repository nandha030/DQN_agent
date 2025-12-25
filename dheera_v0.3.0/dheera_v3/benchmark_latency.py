#!/usr/bin/env python3
"""
Dheera Latency Benchmark
Tests response times with optimization settings
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dheera import Dheera


def benchmark_query(dheera, query, label="Query"):
    """Benchmark a single query"""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Query: \"{query}\"")
    print("-" * 60)

    start = time.time()
    response, metadata = dheera.process_message(query)
    elapsed = (time.time() - start) * 1000

    print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
    print("-" * 60)
    print(f"⏱️  Total Latency:    {elapsed:.0f}ms ({elapsed/1000:.2f}s)")
    print(f"🤖 SLM Latency:      {metadata.get('slm_latency_ms', 0):.0f}ms")
    print(f"📝 Tokens:           {metadata.get('tokens_used', 0)}")
    print(f"🧠 Action:           {metadata.get('action_name', 'N/A')}")
    print(f"💾 RAG Used:         {'Yes' if metadata.get('rag_used') else 'No'}")
    print(f"🔍 Search:           {'Yes' if metadata.get('search_performed') else 'No'}")

    return {
        "label": label,
        "query": query,
        "total_ms": elapsed,
        "slm_ms": metadata.get('slm_latency_ms', 0),
        "tokens": metadata.get('tokens_used', 0),
        "rag_used": metadata.get('rag_used', False),
    }


def main():
    print("=" * 60)
    print("🚀 Dheera Latency Benchmark")
    print("Testing optimized configuration...")
    print("=" * 60)

    # Initialize Dheera
    print("\nInitializing Dheera...")
    dheera = Dheera()
    dheera.start_episode()

    # Test queries (ordered by complexity)
    test_cases = [
        ("Simple Greeting", "Hello!"),
        ("Simple Thanks", "Thanks"),
        ("Short Statement", "I am Nandhavignesh"),
        ("Simple Question", "What is Python?"),
        ("Factual Question", "Explain machine learning"),
        ("Complex Request", "Help me understand neural networks and how they work"),
    ]

    results = []

    for label, query in test_cases:
        result = benchmark_query(dheera, query, label)
        results.append(result)
        time.sleep(0.5)  # Brief pause between queries

    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Test Case':<25} {'Total (ms)':<12} {'SLM (ms)':<12} {'RAG':<6}")
    print("-" * 60)

    total_time = 0
    for r in results:
        total_time += r['total_ms']
        rag_mark = "✓" if r['rag_used'] else "✗"
        print(f"{r['label']:<25} {r['total_ms']:>10.0f}ms  {r['slm_ms']:>10.0f}ms  {rag_mark:>4}")

    avg_time = total_time / len(results)

    print("-" * 60)
    print(f"{'Average Latency':<25} {avg_time:>10.0f}ms")
    print()

    # Performance grades
    print("🎯 PERFORMANCE GRADES:")
    print()

    if avg_time < 3000:
        grade = "A+ (Excellent)"
        emoji = "🌟"
    elif avg_time < 5000:
        grade = "A (Great)"
        emoji = "✨"
    elif avg_time < 10000:
        grade = "B (Good)"
        emoji = "👍"
    elif avg_time < 20000:
        grade = "C (Acceptable)"
        emoji = "🆗"
    else:
        grade = "D (Needs Work)"
        emoji = "⚠️"

    print(f"  Overall: {grade} {emoji}")
    print()

    # Breakdown
    simple_avg = sum(r['total_ms'] for r in results[:3]) / 3
    complex_avg = sum(r['total_ms'] for r in results[3:]) / 3

    print(f"  Simple queries:  {simple_avg:>6.0f}ms avg")
    print(f"  Complex queries: {complex_avg:>6.0f}ms avg")
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print()

    if avg_time > 10000:
        print("  ⚠️  High latency detected!")
        print("  → Check if Ollama is running: `ollama list`")
        print("  → Consider using a faster model: `gemma:2b`")
        print("  → Review LATENCY_OPTIMIZATIONS.md for more tips")
    elif avg_time > 5000:
        print("  ℹ️  Moderate latency")
        print("  → Consider disabling unused modules (goal_evaluator, planner)")
        print("  → See LATENCY_OPTIMIZATIONS.md Level 2")
    else:
        print("  ✅ Great performance!")
        print("  → Current optimizations are working well")
        print("  → No immediate action needed")

    print()

    # Stats
    stats = dheera.get_stats()
    print("=" * 60)
    print("📈 SYSTEM STATS")
    print("=" * 60)
    print()
    print(f"  Conversation turns: {stats['conversation_turns']}")
    print(f"  DQN steps:          {stats['dqn']['total_steps']}")
    print(f"  RAG documents:      {stats['rag']['total_documents']}")
    print(f"  SLM requests:       {stats['slm']['total_requests']}")
    print(f"  SLM avg latency:    {stats['slm']['avg_latency_ms']:.0f}ms")
    print()

    dheera.end_episode("Benchmark complete")

    print("=" * 60)
    print("✅ Benchmark Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
