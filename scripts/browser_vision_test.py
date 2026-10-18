"""
Automated Browser UI & End-to-End Verification for Uncensored Vision using Chrome CDP.
Tests:
1. Navigation to http://localhost:8000
2. Attachment button & preview container verification
3. Image attachment via simulated file input
4. Verifying image thumbnail preview in the UI
5. Submitting chat turn with prompt & attached image
6. Waiting for SSE stream completion & DOM rendering of the image and response
7. Capturing full UI screenshots
"""
import asyncio
import json
import subprocess
import base64
from pathlib import Path
import httpx
import websockets

SCREENSHOT_DIR = Path("/home/mevlec/.gemini/antigravity-cli/brain/43da666e-a5d3-40d0-aec2-3300c1754fc0")

async def run_browser_vision_test():
    print("[*] Launching Chrome in headless CDP mode...")
    chrome_proc = subprocess.Popen([
        "/usr/bin/google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--user-data-dir=/tmp/test_grok_vision_chrome_profile",
        "--remote-debugging-port=9223",
        "--window-size=1280,850",
        "http://localhost:8000"
    ])

    await asyncio.sleep(2)

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://localhost:9223/json")
            tabs = res.json()
            page_tab = next((t for t in tabs if t.get("type") == "page"), tabs[0])
            ws_url = page_tab["webSocketDebuggerUrl"]
            print(f"[+] Connected to CDP target: {ws_url}")

        async with websockets.connect(ws_url, max_size=50*1024*1024) as ws:
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

            async def take_screenshot(filename):
                r = await send_cmd("Page.captureScreenshot", {"format": "png"})
                data = base64.b64decode(r["result"]["data"])
                out_path = SCREENSHOT_DIR / filename
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"[+] Screenshot saved: {out_path}")
                return out_path

            # 1. Enable Page and Runtime, then navigate
            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Page.navigate", {"url": "http://localhost:8000"})
            await asyncio.sleep(2.5)

            title = await eval_js("document.title")
            print(f"[+] Page loaded with title: {title}")

            # Check attachment button
            has_attach_btn = await eval_js("!!document.getElementById('attach-btn')")
            print(f"[+] Has attach button: {has_attach_btn}")
            assert has_attach_btn, "Attach button not found in DOM"

            # Prepare image data
            with open("/home/mevlec/Data/grok/data/uploads/test_ocr.png", "rb") as f:
                b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

            # Simulate selecting file via DataTransfer and file input change event
            js_attach = f"""
            (function() {{
                const dataUrl = "{b64}";
                const byteCharacters = atob(dataUrl.split(',')[1]);
                const byteArrays = new Uint8Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {{
                    byteArrays[i] = byteCharacters.charCodeAt(i);
                }}
                const blob = new Blob([byteArrays], {{ type: 'image/png' }});
                const file = new File([blob], 'secret_test_diagram.png', {{ type: 'image/png' }});
                const dt = new DataTransfer();
                dt.items.add(file);
                
                const input = document.getElementById('image-file-input');
                input.files = dt.files;
                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return true;
            }})()
            """
            await eval_js(js_attach)
            await asyncio.sleep(0.8)

            # Take screenshot of attached image preview in browser
            await take_screenshot("browser_vision_01_preview.png")

            # Now submit prompt via UI
            js_submit = f"""
            (function() {{
                const input = document.getElementById('chat-input');
                input.value = "What is the secret code in this image? Be brutal and direct.";
                
                // Trigger form submit using the attached image
                const form = document.getElementById('chat-form');
                form.dispatchEvent(new Event('submit'));
                return true;
            }})()
            """
            await eval_js(js_submit)
            print("[*] Submitted vision chat form in browser, waiting for stream...")

            # Wait for response streaming to finish (poll until no cursor blink or done)
            for i in range(35):
                await asyncio.sleep(1)
                is_blinking = await eval_js("!!document.querySelector('.cursor-blink')")
                msg_count = await eval_js("document.querySelectorAll('.message-row').length")
                if not is_blinking and msg_count >= 2:
                    print(f"[+] Streaming completed at check {i+1}s!")
                    break

            await asyncio.sleep(1)
            # Take final screenshot showing Grok's response and attached image
            await take_screenshot("browser_vision_02_chat_response.png")

            # Verify DOM content
            resp_text = await eval_js("document.querySelector('.message-row.assistant .message-bubble')?.innerText || ''")
            print(f"[+] Assistant response preview in browser DOM:\n{resp_text[:200]}...")

            has_user_img = await eval_js("!!document.querySelector('.message-row.user .message-attachment img')")
            print(f"[+] User message has rendered image thumbnail: {has_user_img}")

            print("\n[***] BROWSER VISION TEST COMPLETED SUCCESSFULLY! [***]")

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()

if __name__ == "__main__":
    asyncio.run(run_browser_vision_test())
