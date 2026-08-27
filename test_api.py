"""Quick test script for the /evaluate endpoint."""
import httpx

resp = httpx.post(
    "http://localhost:8000/evaluate",
    json={"prompt": "Explain Kubernetes to a junior developer."},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()

print("=" * 75)
print(f"{'Model':<15} | {'Avg Score':>9} | {'Latency':>8} | {'Cost':>10} | {'Tokens':>6}")
print("-" * 75)
for d in data:
    avg = (d["correctness_score"] + d["relevance_score"] + d["faithfulness_score"]) / 3
    print(f"{d['model_name']:<15} | {avg:>9.4f} | {d['latency_seconds']:>7.3f}s | ${d['estimated_cost_usd']:>8.6f} | {d['total_tokens']:>6}")
print("=" * 75)
print("\nTest PASSED!")
