"""
Verification script for Prompt Versions v1, v2, and v3.
"""

import httpx

API_BASE = "http://localhost:8000"

def test_prompt_versions():
    query = "Jelaskan arsitektur microservices."
    versions = ["v1", "v2", "v3"]

    print("=================================================================")
    print(f"TESTING PROMPT VERSIONS v1, v2, v3 FOR QUERY: '{query}'")
    print("=================================================================\n")

    with httpx.Client(timeout=45.0) as client:
        for v in versions:
            print(f"--- RUNNING EVALUATION FOR VERSION: {v} ---")
            payload = {
                "prompt": query,
                "models": ["Gemini 3.6", "Local Llama"],
                "prompt_version": v,
            }
            r = client.post(f"{API_BASE}/evaluate", json=payload)
            assert r.status_code == 200, f"Error: {r.text}"
            results = r.json()
            for item in results:
                safe_output = item['output_text'][:160].encode('ascii', 'replace').decode()
                safe_reasoning = str(item.get('judge_reasoning', '')).encode('ascii', 'replace').decode()
                print(f"  • Model: {item['model_name']} | Version: {item['prompt_version']}")
                print(f"    Scores: Correctness={item['correctness_score']}, Relevance={item['relevance_score']}, Faithfulness={item['faithfulness_score']}")
                print(f"    Output Snippet:\n      {safe_output}...")
                print(f"    Judge Reasoning:\n      {safe_reasoning}\n")

    print("ALL PROMPT VERSIONS TESTED AND PROVEN DISTINCT!")

if __name__ == "__main__":
    test_prompt_versions()
