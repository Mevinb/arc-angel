"""Download tool — save generated images to ARC's download folder."""

from __future__ import annotations

import hashlib
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from ..safety.permissions import RiskLevel
from .base import Tool, ToolResult


class DownloadTool(Tool):
    name = "file.download"
    description = (
        "Download a file from a URL to ARC's download folder (data/downloads). "
        "Use this to save generated images from ChatGPT (cdn.oaistatic.com, files.oaiusercontent.com) "
        "or any image URL. Returns the local path."
    )
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "url": {"type": "string", "description": "URL to download"},
            "filename": {"type": "string", "description": "Optional filename (e.g. beach-girl.png). Auto-generated if empty."},
            "subfolder": {"type": "string", "description": "Subfolder inside downloads (e.g. images, chatgpt). Default: images"},
        },
        "required": ["url"],
    }

    def __init__(self, download_dir: Optional[Path] = None) -> None:
        # Default to data/downloads relative to project root
        if download_dir is None:
            # Try to resolve from config if available, else fallback
            try:
                from ..config import load_config
                cfg = load_config()
                download_dir = cfg.data_dir / "downloads"
            except Exception:
                download_dir = Path("data/downloads")
        self.download_dir = Path(download_dir)

    def run(self, url: str = "", filename: str = "", subfolder: str = "images", **_: Any) -> ToolResult:
        url = (url or "").strip()
        if not url:
            return ToolResult.failure("No URL provided")
        if not url.startswith(("http://", "https://")):
            return ToolResult.failure(f"Invalid URL: {url}")

        # Sanitize subfolder
        subfolder = (subfolder or "images").strip().strip("/")
        # Prevent path traversal
        if ".." in subfolder or subfolder.startswith("/"):
            subfolder = "images"
        # Only allow alphanumeric, -, _
        subfolder = re.sub(r"[^a-zA-Z0-9._-]", "_", subfolder)[:50] or "images"

        # Determine filename
        if filename:
            filename = Path(filename).name  # strip dirs
            filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        else:
            # Try to infer from URL
            parsed = urlparse(url)
            name = Path(parsed.path).name
            if name and "." in name:
                filename = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
            else:
                # Generate from URL hash + content type
                h = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"image-{int(time.time())}-{h}.png"

        # Ensure extension
        if "." not in filename:
            # Try to guess from URL or content-type after fetch
            filename += ".png"

        target_dir = self.download_dir / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        # Avoid overwrite: add suffix if exists
        counter = 1
        base = target.stem
        suffix = target.suffix
        while target.exists():
            target = target_dir / f"{base}-{counter}{suffix}"
            counter += 1
            if counter > 100:
                break

        try:
            resp = requests.get(url, stream=True, timeout=30, headers={"User-Agent": "ARC/1.0"})
            resp.raise_for_status()
            # Try to update filename from content-type if needed
            ctype = resp.headers.get("content-type", "")
            if "image" in ctype:
                ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or suffix
                if not target.suffix or target.suffix == ".bin":
                    target = target.with_suffix(ext)
            # Also check content-disposition
            cd = resp.headers.get("content-disposition", "")
            if "filename=" in cd:
                m = re.search(r'filename="?([^"]+)"?', cd)
                if m:
                    cd_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(m.group(1)).name)
                    if cd_name:
                        target = target_dir / cd_name

            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            size = target.stat().st_size
            return ToolResult.success(
                f"Downloaded {url} → {target} ({size} bytes, {ctype or 'unknown type'})",
                url=url,
                path=str(target),
                size=size,
            )
        except Exception as exc:
            return ToolResult.failure(f"Download failed for {url}: {exc}")
