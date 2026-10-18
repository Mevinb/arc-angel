"""
Test script to evaluate conversational naturalness, pacing, and human-like flow.
"""
import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_conversation():
    print("=== Testing Natural Conversational Flow ===")
    
    # Create test session
    req = urllib.request.Request(
        f"{BASE_URL}/api/sessions",
        data=json.dumps({"title": "Natural Flow Test", "mode": "classic"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        sess = json.loads(resp.read().decode())
    session_id = sess["id"]
    print(f"[+] Session created: {session_id}\n")

    test_prompts = [
        ("Casual Chit-Chat", "hey grok, how's it going?"),
        ("Burnout & Empathy", "man, I've been feeling pretty burned out lately with all the med prep... feel like I'm running in circles"),
        ("Opinion & Banter", "be real with me, is pineapple on pizza actually a crime or are people just dramatic?")
    ]

    for label, prompt in test_prompts:
        print(f"--- [User] ({label}): \"{prompt}\" ---")
        payload = {
            "session_id": session_id,
            "message": prompt,
            "mode": "classic"
        }
        req = urllib.request.Request(
            f"{BASE_URL}/api/chat/stream",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )

        full_reply = []
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line_str = line.decode().strip()
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    if data.get("type") == "token":
                        full_reply.append(data.get("token", ""))

        reply_text = "".join(full_reply).strip()
        print(f"[Grok]:\n{reply_text}\n")
        
        # Quality assertions
        has_bullets = any(line.strip().startswith(("-", "*", "1.", "2.")) for line in reply_text.split("\n"))
        print(f"[Check] Unprompted bullets present: {has_bullets}")
        print(f"[Check] Word count: {len(reply_text.split())} words\n")
        time.sleep(1)

if __name__ == "__main__":
    test_conversation()
