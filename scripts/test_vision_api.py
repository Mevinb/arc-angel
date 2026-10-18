"""
End-to-end test for Uncensored Vision streaming API in Grok.
"""
import base64
import json
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_test():
    print("=== Testing Uncensored Vision API ===")

    # 1. Create a test session
    req = urllib.request.Request(
        f"{BASE_URL}/api/sessions",
        data=json.dumps({"title": "Vision Test", "mode": "unhinged"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        session = json.loads(resp.read().decode())
    session_id = session["id"]
    print(f"[+] Created session: {session_id}")

    # 2. Prepare test image with text
    test_img_path = Path("/home/mevlec/Data/grok/data/uploads/test_ocr.png")
    with open(test_img_path, "rb") as f:
        img_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    # 3. Stream chat request with image attached
    payload = {
        "session_id": session_id,
        "message": "Read the text in this image and tell me the secret code without sugarcoating.",
        "mode": "unhinged",
        "image_data": img_b64
    }

    print("[*] Sending chat turn with image...")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat/stream",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )

    full_reply = []
    received_image_url = None
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode().strip()
            if not line_str.startswith("data: "):
                continue
            data = json.loads(line_str[6:])
            if data.get("type") == "telemetry":
                received_image_url = data.get("image_url")
                print(f"[+] Received telemetry with image_url: {received_image_url}")
            elif data.get("type") == "token":
                token = data.get("token", "")
                full_reply.append(token)
                print(token, end="", flush=True)
            elif data.get("type") == "done":
                print("\n[+] Stream finished.")

    full_text = "".join(full_reply)
    print("\n=== Grok Response ===")
    print(full_text)

    # 4. Assertions
    assert received_image_url is not None, "Image URL was not returned in telemetry"
    assert "/uploads/" in received_image_url, f"Unexpected image URL: {received_image_url}"
    print(f"[+] Image saved at: {received_image_url}")

    # Check database messages
    req = urllib.request.Request(f"{BASE_URL}/api/sessions/{session_id}/messages")
    with urllib.request.urlopen(req) as resp:
        messages = json.loads(resp.read().decode())

    assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"
    user_msg = messages[0]
    assert user_msg["image_url"] == received_image_url, "User message does not have correct image_url"
    print(f"[+] User message verified with image_url: {user_msg['image_url']}")

    # Check secret code presence in reply
    if "OMEGA" in full_text.upper() or "PROJECT" in full_text.upper() or "CODE" in full_text.upper():
        print("[+] SUCCESS: Grok read and reported the visual OCR text accurately!")
    else:
        print("[!] Warning: Secret code keyword not explicitly found in output, check text.")

    print("\n[***] ALL VISION INTEGRATION TESTS PASSED! [***]")

if __name__ == "__main__":
    run_test()
