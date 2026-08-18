import os
import json
import base64
import requests
import time
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path, max_size=(512, 512)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def call_recap_api(video_info, keyframes_paths, subtitle, previous_memory, template_name="prompt_template.txt", model=None):
    api_key = os.getenv("RECAP_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("RECAP_API_KEY / OPENROUTER_API_KEY environment variable not set")
        
    api_base = os.getenv("RECAP_API_BASE", "https://api.apimaster.ai/v1/chat/completions")
    if model is None:
        model = os.getenv("RECAP_MODEL", "gpt-5.4-mini")
        
    template_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'api', template_name)
    if not os.path.exists(template_path):
        template_path = os.path.join(os.getcwd(), 'api', template_name)
    with open(template_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
        
    content = [
        {"type": "text", "text": f"Video Info:\n{video_info}\n\nPrevious Memory:\n{previous_memory}\n\nSubtitle:\n{subtitle}\n\nKeyframes:"}
    ]
    
    for path in keyframes_paths:
        base64_img = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
        
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1000
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(5):
        try:
            response = requests.post(
                api_base,
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code in [429, 502, 503, 504]:
                print(f"API HTTP {response.status_code}, retrying attempt {attempt+1}/5...")
                time.sleep(3 ** attempt)
                continue
                
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                
            response.raise_for_status()
            result = response.json()
            
            try:
                output_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not output_text:
                    raise ValueError("Empty or null content in response")
                return json.loads(output_text)
            except (KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
                safe_text = repr(output_text)[:200] if output_text else "None"
                safe_text = safe_text.encode('ascii', 'backslashreplace').decode('ascii')
                print(f"Error parsing response JSON: {safe_text}")
                if attempt < 4:
                    time.sleep(2)
                    continue
                # If we fail 5 times, return a dummy so the video isn't skipped entirely
                print("Max retries for JSON parsing hit, returning fallback.")
                return {"caption": "API failed to generate caption.", "memory": previous_memory}
                
        except requests.exceptions.RequestException as e:
            if attempt == 4:
                print(f"API Request failed 5 times: {e}")
                return {"caption": "API request failed.", "memory": previous_memory}
            time.sleep(3 ** attempt)
            
    raise Exception("Max retries exceeded")

if __name__ == "__main__":
    pass
