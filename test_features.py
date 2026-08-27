"""
Verification script for LLM-as-a-Judge, Ground Truth, and Prompt A/B evaluation.
"""

import httpx

API_BASE = "http://localhost:8000"

def test_system():
    print("Testing /health endpoint...")
    with httpx.Client() as client:
        r = client.get(f"{API_BASE}/health")
        assert r.status_code == 200, f"Health failed: {r.text}"
        print("  -> Health OK:", r.json())

    print("\nTesting /evaluate endpoint with Ground Truth and AI Judge...")
    payload = {
        "prompt": "Jelaskan apa itu CI/CD dalam software development.",
        "models": ["Gemini 3.6", "Local Llama"],
        "prompt_version": "v1",
        "ground_truth": "CI/CD adalah Continuous Integration dan Continuous Delivery/Deployment untuk otomatisasi build, test, dan release.",
    }
    with httpx.Client(timeout=40.0) as client:
        r = client.post(f"{API_BASE}/evaluate", json=payload)
        assert r.status_code == 200, f"Evaluate failed: {r.text}"
        data = r.json()
        print(f"  -> Returned {len(data)} evaluated models:")
        for item in data:
            print(f"     • Model: {item['model_name']}")
            print(f"       Score: Correctness={item['correctness_score']}, Relevance={item['relevance_score']}, Faithfulness={item['faithfulness_score']}")
            print(f"       Judge Reasoning: {item.get('judge_reasoning')}")
            print(f"       Latency: {item['latency_seconds']}s | Cost: ${item['estimated_cost_usd']}")

    print("\nTesting /results endpoint...")
    with httpx.Client() as client:
        r = client.get(f"{API_BASE}/results?limit=5")
        assert r.status_code == 200
        history = r.json()
        print(f"  -> History records retrieved: {len(history)}")
        if history:
            print(f"     Latest record: Model={history[0]['model_name']}, GroundTruth={history[0].get('ground_truth')}")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! No bugs detected.")

if __name__ == "__main__":
    test_system()
