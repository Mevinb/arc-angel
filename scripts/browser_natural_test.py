"""
Browser CDP script to capture screenshot of natural conversational flow.
"""
import asyncio
import json
import subprocess
import base64
from pathlib import Path
import httpx
import websockets

SCREENSHOT_DIR = Path("/home/mevlec/.gemini/antigravity-cli/brain/43da666e-a5d3-40d0-aec2-3300c1754fc0")

async def run_browser_capture():
    chrome_proc = subprocess.Popen([
        "/usr/bin/google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--user-data-dir=/tmp/test_grok_natural_chrome_profile",
        "--remote-debugging-port=9224",
        "--window-size=1280,850",
        "http://localhost:8000"
    ])

    await asyncio.sleep(2)

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://localhost:9224/json")
            tabs = res.json()
            page_tab = next((t for t in tabs if t.get("type") == "page"), tabs[0])
            ws_url = page_tab["webSocketDebuggerUrl"]

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

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Page.navigate", {"url": "http://localhost:8000"})
            await asyncio.sleep(2.5)

            # Click on 'Natural Flow Test' session in sidebar if available
            js_click_session = """
            (function() {
                const items = Array.from(document.querySelectorAll('.session-item'));
                const target = items.find(el => el.innerText.includes('Natural Flow Test'));
                if (target) {
                    target.click();
                    return true;
                }
                return false;
            })()
            """
            clicked = await eval_js(js_click_session)
            print(f"[+] Clicked Natural Flow Test session: {clicked}")
            await asyncio.sleep(1.5)

            await take_screenshot("browser_natural_conversation.png")
            print("[+] Capture completed successfully!")

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()

if __name__ == "__main__":
    asyncio.run(run_browser_capture())
