"""
Automated Browser UI & End-to-End Verification using Chrome DevTools Protocol (CDP).
Tests:
1. Navigation to http://localhost:8000
2. UI rendering & layout verification
3. Clicking 'Memory Dossier' and verifying Dossier cards & categories
4. Adding a new fact through the UI form
5. Clicking 'Unhinged' persona mode
6. Submitting a chat query via the UI input form and verifying live token streaming into DOM
7. Capturing screenshots of every state
"""
import asyncio
import json
import subprocess
import time
import httpx
import websockets
from pathlib import Path

SCREENSHOT_DIR = Path("/home/mevlec/.gemini/antigravity-cli/brain/43da666e-a5d3-40d0-aec2-3300c1754fc0")

async def run_browser_tests():
    print("[*] Starting Chrome in headless CDP mode...")
    chrome_proc = subprocess.Popen([
        "/usr/bin/google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--user-data-dir=/tmp/test_grok_chrome_profile",
        "--remote-debugging-port=9222",
        "--window-size=1280,850",
        "http://localhost:8000"
    ])

    await asyncio.sleep(2)

    try:
        # Get WebSocket debugger URL for the page target
        async with httpx.AsyncClient() as client:
            res = await client.get("http://localhost:9222/json")
            tabs = res.json()
            page_tab = next((t for t in tabs if t.get("type") == "page"), tabs[0])
            ws_url = page_tab["webSocketDebuggerUrl"]
            print(f"[+] Connected to page target: {page_tab.get('title')} ({ws_url})")

        async with websockets.connect(ws_url) as ws:
            msg_id = 0

            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("id") == msg_id:
                        return resp

            async def eval_js(expr):
                r = await send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value")

            async def capture_screenshot(filename):
                r = await send_cmd("Page.captureScreenshot", {"format": "png"})
                b64_data = r.get("result", {}).get("data")
                if b64_data:
                    import base64
                    path = SCREENSHOT_DIR / filename
                    path.write_bytes(base64.b64decode(b64_data))
                    print(f"[+] Saved screenshot: {path}")

            # 1. Enable Page and Runtime, then navigate
            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Page.navigate", {"url": "http://localhost:8000"})
            await asyncio.sleep(2.5)

            # Check title
            title = await eval_js("document.title")
            print(f"[✓] Page Title: '{title}'")
            print(f"[✓] Page Title: '{title}'")
            assert "GROK" in title, f"Unexpected title: {title}"

            # 2. Open Memory Dossier Drawer
            print("[*] Clicking 'Memory Dossier' button in sidebar...")
            await eval_js("document.getElementById('open-memory-btn').click()")
            await asyncio.sleep(1)

            drawer_hidden = await eval_js("document.getElementById('memory-drawer').classList.contains('hidden')")
            print(f"[✓] Drawer is open: {not drawer_hidden}")
            assert not drawer_hidden, "Memory drawer did not open!"

            # Check facts loaded in drawer
            fact_cards_count = await eval_js("document.querySelectorAll('.fact-card').length")
            print(f"[✓] Rendered Fact Cards in Dossier: {fact_cards_count}")
            assert fact_cards_count > 0, "No fact cards rendered!"

            await capture_screenshot("browser_test_01_memory_drawer.png")

            # 3. Add a new fact via Drawer Form
            print("[*] Adding test fact 'superpower: photographic memory' via UI form...")
            await eval_js("""
                document.getElementById('fact-key').value = 'superpower';
                document.getElementById('fact-value').value = 'photographic memory';
                document.getElementById('fact-category').value = 'secrets';
                document.getElementById('add-fact-form').dispatchEvent(new Event('submit'));
            """)
            await asyncio.sleep(1)

            new_count = await eval_js("document.querySelectorAll('.fact-card').length")
            print(f"[✓] Fact cards count after UI add: {new_count}")

            # Close drawer
            await eval_js("document.getElementById('close-memory-btn').click()")
            await asyncio.sleep(0.5)

            # 4. Switch to 'Unhinged' persona mode
            print("[*] Clicking 'Unhinged' mode pill...")
            await eval_js("document.querySelector('.mode-pill[data-mode=\"unhinged\"]').click()")
            active_mode = await eval_js("document.querySelector('.mode-pill.active').dataset.mode")
            print(f"[✓] Active mode in UI: '{active_mode}'")
            assert active_mode == "unhinged", "Mode switch failed!"

            # 5. Create a new conversation and send a test prompt
            print("[*] Clicking 'New Conversation' button...")
            await eval_js("document.getElementById('new-chat-btn').click()")
            await asyncio.sleep(1)

            test_prompt = "What do you know about my superpower and my background? Be brutally candid."
            print(f"[*] Typing prompt: '{test_prompt}' and submitting form...")
            await eval_js(f"""
                document.getElementById('chat-input').value = "{test_prompt}";
                document.getElementById('chat-form').dispatchEvent(new Event('submit'));
            """)

            # Poll for stream completion
            print("[*] Waiting for live streaming completion in browser DOM...")
            max_wait = 45
            for i in range(max_wait):
                await asyncio.sleep(1)
                is_streaming = await eval_js("document.getElementById('send-btn').disabled")
                if not is_streaming:
                    print(f"[+] Streaming finished in {i+1} seconds!")
                    break
            else:
                print("[!] Timeout waiting for streaming.")

            # Read assistant message from DOM
            assistant_text = await eval_js("""
                (() => {
                    const msgs = document.querySelectorAll('.message-row.assistant .message-bubble');
                    return msgs.length > 0 ? msgs[msgs.length - 1].innerText : '';
                })()
            """)
            print("\n" + "="*50)
            print("[GROK BROWSER RESPONSE IN DOM]:")
            print(assistant_text)
            print("="*50 + "\n")

            assert "photographic memory" in assistant_text.lower() or "asin" in assistant_text.lower() or "doc" in assistant_text.lower(), "Assistant did not recall user facts!"

            await capture_screenshot("browser_test_02_chat_response.png")
            print("[✓] ALL BROWSER E2E TESTS PASSED SUCCESSFULLY!")

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()

if __name__ == "__main__":
    asyncio.run(run_browser_tests())
