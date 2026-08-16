import time
import requests
import json
import os
import sys

def test_api_latency():
    print("Testing OpenRouter API latency for xiaomi/mimo-v2.5...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not found in environment.")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "xiaomi/mimo-v2.5",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Output JSON."
            },
            {
                "role": "user",
                "content": "Hello, please reply with a JSON with a 'message' field."
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"Success! Latency: {end_time - start_time:.2f} seconds")
            print("Response length:", len(response.text))
        else:
            print(f"Error {response.status_code}: {response.text}")
            print(f"Failed after {end_time - start_time:.2f} seconds")
            
    except Exception as e:
        end_time = time.time()
        print(f"Exception after {end_time - start_time:.2f} seconds: {e}")

if __name__ == "__main__":
    test_api_latency()
