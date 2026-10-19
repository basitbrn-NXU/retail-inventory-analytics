"""Exercise a running HTTP endpoint and verify the frozen example forecast."""
import argparse
import json
import numpy as np
import httpx
from core import read_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    manifest = read_json("registry.json")
    base = args.url.rstrip("/")
    with httpx.Client(trust_env=False, timeout=20) as client:
        health = client.get(base + "/health")
        health.raise_for_status()
        response = client.post(base + "/predict", json=manifest["example_request"])
        response.raise_for_status()
    output = response.json()
    if not np.isclose(output["predictions"][0], manifest["example_prediction"], rtol=1e-10):
        raise ValueError("API prediction differs from Module 4 example.")
    print(json.dumps({"health": health.json(), "prediction": output}, indent=2))

if __name__ == "__main__":
    main()
