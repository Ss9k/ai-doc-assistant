import httpx

url = "http://127.0.0.1:8000/ask"

payload = {
    "question": "Explain Python decorators in simple terms",
    "provider": "ollama",
}

with httpx.stream(
    "POST",
    url,
    json=payload,
    timeout=120,
) as response:
    print("Status:", response.status_code)
    print("Streaming response:")

    response.raise_for_status()

    for chunk in response.iter_text():
        print(chunk, end="", flush=True)

print("\n\nStreaming completed.")
