"""
Parse a T-Mobile InfoBot curl command and extract:
  - Bearer Token
  - CONVERSATION_ID
  - SESSION_ID
  - INTERACTION_ID

Usage:
  python parse_curl.py                  # reads from stdin
  python parse_curl.py curl.txt         # reads from file
  echo "<curl>" | python parse_curl.py
"""

import re
import json
import sys
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:latest"


def extract_with_regex(curl_text: str) -> dict:
    """Fast regex path — works for standard curl formats."""
    result = {}

    # Bearer token from Authorization header
    bearer = re.search(r"-H\s+['\"]authorization:\s+Bearer\s+([^\s'\"]+)", curl_text, re.IGNORECASE)
    if bearer:
        result["BEARER_TOKEN"] = bearer.group(1)

    # session-id header
    session = re.search(r"-H\s+['\"]session-id:\s+([^'\"]+)['\"]", curl_text, re.IGNORECASE)
    if session:
        result["SESSION_ID"] = session.group(1).strip()

    # interaction-id header
    interaction = re.search(r"-H\s+['\"]interaction-id:\s+([^'\"]+)['\"]", curl_text, re.IGNORECASE)
    if interaction:
        result["INTERACTION_ID"] = interaction.group(1).strip()

    # conversationId in --data-raw JSON body
    conv = re.search(r'"conversationId"\s*:\s*"([^"]+)"', curl_text)
    if conv:
        result["CONVERSATION_ID"] = conv.group(1)

    return result


def extract_with_llm(curl_text: str) -> dict:
    """Fallback: ask local Gemma4 via Ollama."""
    prompt = f"""Extract exactly these four values from the curl command below and return ONLY valid JSON with these keys: BEARER_TOKEN, CONVERSATION_ID, SESSION_ID, INTERACTION_ID.

Curl command:
{curl_text}

Return only the JSON object, no explanation."""

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    raw = body.get("response", "")
    # Pull the JSON block out of the response
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM returned no JSON object:\n{raw}")
    return json.loads(match.group(0))


def parse(curl_text: str) -> dict:
    result = extract_with_regex(curl_text)

    missing = [k for k in ("BEARER_TOKEN", "CONVERSATION_ID", "SESSION_ID", "INTERACTION_ID") if k not in result]
    if missing:
        print(f"Regex missed {missing}, falling back to LLM...", file=sys.stderr)
        llm_result = extract_with_llm(curl_text)
        for k in missing:
            if k in llm_result:
                result[k] = llm_result[k]

    return result


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            curl_text = f.read()
    else:
        print("Paste curl command (Ctrl+D when done):", file=sys.stderr)
        curl_text = sys.stdin.read()

    result = parse(curl_text)

    print("\n--- Extracted Fields ---")
    for key in ("BEARER_TOKEN", "CONVERSATION_ID", "SESSION_ID", "INTERACTION_ID"):
        value = result.get(key, "<NOT FOUND>")
        display = value if len(value) < 80 else value[:40] + "..." + value[-20:]
        print(f"{key}: {display}")

    print("\n--- Copy-Paste Block ---")
    for key in ("BEARER_TOKEN", "CONVERSATION_ID", "SESSION_ID", "INTERACTION_ID"):
        value = result.get(key, "")
        print(f'{key}="{value}"')

    json_str = json.dumps(result, indent=2)
    try:
        import subprocess
        subprocess.run("pbcopy", input=json_str.encode(), check=True)
        print("\nJSON copied to clipboard.")
    except Exception:
        pass

    print("\n--- Full JSON ---")
    print(json_str)


if __name__ == "__main__":
    main()
