"""
Intelligent Browser & Logged-In Profile Discovery for Linux (Wayland / X11 / Xvfb).
Inspects local Chromium, Brave, and Chrome installations, parses Local State
profile.info_cache, analyzes session cookies and activity timestamps, and maps
user-friendly names (e.g. "mevinbenty", "Personal") to internal directories ("Default", "Profile 2").
"""
import os
import re
import json
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger("browser_profiles")


@dataclass
class BrowserProfileInfo:
    browser_name: str         # 'chrome', 'brave', 'chromium'
    binary_cmd: str           # 'google-chrome', '/snap/bin/brave', 'brave-browser'
    user_data_dir: Path       # Path to directory containing Local State
    profile_dir: str          # 'Default', 'Profile 1', 'Profile 2'
    display_name: str         # 'Mevin', 'Mevin Benty', 'Personal'
    user_name: str            # 'mevinbenty507@gmail.com' or ''
    gaia_name: Optional[str]  # 'Mevin Benty' or None
    is_logged_in: bool        # True if account is connected or significant cookie store
    cookies_size: int         # Size of Cookies SQLite database in bytes
    last_active_ts: float     # Timestamp of last activity


# Candidate Linux browser configurations
BROWSER_CANDIDATES = [
    {
        "name": "chrome",
        "bins": ["google-chrome", "google-chrome-stable"],
        "paths": [
            Path.home() / ".config/google-chrome",
            Path.home() / ".var/app/com.google.Chrome/config/google-chrome",
        ]
    },
    {
        "name": "brave",
        "bins": ["/snap/bin/brave", "brave-browser", "brave"],
        "paths": [
            Path.home() / "snap/brave/current/.config/BraveSoftware/Brave-Browser",
            Path.home() / ".config/BraveSoftware/Brave-Browser",
            Path.home() / ".var/app/com.brave.Browser/config/BraveSoftware/Brave-Browser",
        ]
    },
    {
        "name": "chromium",
        "bins": ["chromium", "chromium-browser", "/snap/bin/chromium"],
        "paths": [
            Path.home() / ".config/chromium",
            Path.home() / "snap/chromium/current/.config/chromium",
        ]
    },
]


def _find_binary(bins: List[str]) -> str:
    """Finds an executable binary from candidate list, or falls back to first entry."""
    for b in bins:
        if shutil.which(b) or (os.path.isabs(b) and os.path.exists(b)):
            return b
    return bins[0]


def discover_browser_profiles() -> List[BrowserProfileInfo]:
    """
    Scans Linux user config directories for installed Chromium-based browsers,
    parses Local State to extract profile metadata, checks cookie volumes and
    modification times, and returns a prioritized list sorted by login presence,
    cookies volume, and recent activity.
    """
    profiles: List[BrowserProfileInfo] = []
    seen_dirs = set()

    for candidate in BROWSER_CANDIDATES:
        bname = candidate["name"]
        binary_cmd = _find_binary(candidate["bins"])

        for udir in candidate["paths"]:
            if not udir.exists():
                continue

            local_state_file = udir / "Local State"
            if not local_state_file.exists():
                continue

            resolved_key = str(udir.resolve())
            if resolved_key in seen_dirs:
                continue
            seen_dirs.add(resolved_key)

            try:
                with open(local_state_file, "r", encoding="utf-8", errors="ignore") as f:
                    state_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error parsing Local State in {udir}: {e}")
                continue

            info_cache = state_data.get("profile", {}).get("info_cache", {})
            if not info_cache:
                # Minimal fallback: check if Default folder exists
                if (udir / "Default").exists():
                    info_cache = {"Default": {"name": "Default", "user_name": ""}}

            for pdir, pdata in info_cache.items():
                pfolder = udir / pdir
                if not pfolder.exists():
                    continue

                display_name = (pdata.get("name") or pdata.get("gaia_name") or pdir).strip()
                user_name = (pdata.get("user_name") or "").strip()
                gaia_name = (pdata.get("gaia_name") or "").strip() or None

                # Locate Cookies database
                cookies_file = pfolder / "Cookies"
                if not cookies_file.exists():
                    cookies_file = pfolder / "Network" / "Cookies"

                cookies_size = 0
                cookies_mtime = 0.0
                if cookies_file.exists():
                    try:
                        stat = cookies_file.stat()
                        cookies_size = stat.st_size
                        cookies_mtime = stat.st_mtime
                    except Exception:
                        pass

                # History mtime
                history_file = pfolder / "History"
                history_mtime = 0.0
                if history_file.exists():
                    try:
                        history_mtime = history_file.stat().st_mtime
                    except Exception:
                        pass

                active_time = float(pdata.get("active_time") or 0.0)
                # Chromium active_time is sometimes in webkit/unix epochs; keep max plausible unix timestamp
                last_active_ts = max(active_time if active_time < 3000000000 else 0.0, cookies_mtime, history_mtime)

                # Profile is considered logged in if it has an account email/gaia_id, or substantial auth cookies (>30KB)
                has_account = bool(user_name or pdata.get("gaia_id") or pdata.get("hosted_domain"))
                is_logged_in = has_account or (cookies_size > 30000)

                profiles.append(
                    BrowserProfileInfo(
                        browser_name=bname,
                        binary_cmd=binary_cmd,
                        user_data_dir=udir,
                        profile_dir=pdir,
                        display_name=display_name,
                        user_name=user_name,
                        gaia_name=gaia_name,
                        is_logged_in=is_logged_in,
                        cookies_size=cookies_size,
                        last_active_ts=last_active_ts,
                    )
                )

    # Sort profiles: logged-in first, then largest cookies (more active sessions), then most recent activity
    profiles.sort(
        key=lambda p: (
            1 if p.is_logged_in else 0,
            1 if p.user_name else 0,
            p.cookies_size,
            p.last_active_ts
        ),
        reverse=True
    )
    return profiles


def get_available_profiles_summary() -> str:
    """
    Returns a concise text summary of all installed browsers and their active profiles
    suitable for direct prompt injection into DesktopAgent's system prompt.
    """
    profiles = discover_browser_profiles()
    if not profiles:
        return "No local browser profiles detected."

    by_browser: Dict[str, List[BrowserProfileInfo]] = {}
    for p in profiles:
        by_browser.setdefault(p.browser_name, []).append(p)

    lines = ["Available Logged-In Browser Profiles:"]
    for bname, plist in by_browser.items():
        desc_parts = []
        for p in plist[:5]:  # limit to top 5 profiles per browser
            status_tags = []
            if p.is_logged_in:
                status_tags.append("logged-in")
            if p.user_name:
                status_tags.append(p.user_name)

            tag_str = f" [{', '.join(status_tags)}]" if status_tags else ""
            desc_parts.append(f'"{p.display_name}"{tag_str} (dir: {p.profile_dir})')

        lines.append(f"- {bname}: " + ", ".join(desc_parts))

    return "\n".join(lines)


def resolve_browser_and_profile(
    requested_browser: Optional[str] = None,
    requested_profile: Optional[str] = None
) -> Tuple[str, str, Path]:
    """
    Resolves requested browser and profile names to an exact executable binary,
    internal profile folder on disk ('Default', 'Profile 1', etc.), and user_data_dir.

    Examples:
      resolve_browser_and_profile("chrome", "mevinbenty") -> ('google-chrome', 'Default', Path('~/.config/google-chrome'))
      resolve_browser_and_profile("brave", "mevin")        -> ('/snap/bin/brave', 'Profile 2', Path('~/snap/brave/...'))
      resolve_browser_and_profile(None, None)              -> top active logged-in profile on system!
    """
    profiles = discover_browser_profiles()
    if not profiles:
        # Fallback when no Chromium profile discovered
        bin_cmd = "google-chrome" if shutil.which("google-chrome") else "brave-browser"
        return bin_cmd, "Default", Path.home() / ".config/google-chrome"

    # Filter by browser if requested
    filtered = profiles
    if requested_browser:
        req_b = requested_browser.strip().lower()
        if "chrome" in req_b:
            filtered = [p for p in profiles if p.browser_name == "chrome"] or profiles
        elif "brave" in req_b:
            filtered = [p for p in profiles if p.browser_name == "brave"] or profiles
        elif "chromium" in req_b:
            filtered = [p for p in profiles if p.browser_name == "chromium"] or profiles

    # If no profile requested, pick the top-ranked profile in the filtered list
    if not requested_profile:
        top = filtered[0]
        return top.binary_cmd, top.profile_dir, top.user_data_dir

    # Fuzzy match requested profile
    query = requested_profile.strip().lower()
    best_profile: Optional[BrowserProfileInfo] = None
    best_score = -1.0

    for p in filtered:
        score = 0.0
        pdir_low = p.profile_dir.lower()
        disp_low = p.display_name.lower()
        user_low = p.user_name.lower()
        gaia_low = (p.gaia_name or "").lower()

        # 1. Exact match on profile directory ('default', 'profile 2')
        if query == pdir_low:
            score = 100.0
        # 2. Exact match on display name
        elif query == disp_low:
            score = 95.0
        # 3. Exact match on username/email
        elif query == user_low or (user_low and query == user_low.split("@")[0]):
            score = 90.0
        # 4. Query is a substring of username/email (e.g. 'mevinbenty' in 'mevinbenty507@gmail.com')
        elif query in user_low:
            score = 85.0
        # 5. Query is a substring of display name (e.g. 'mevin' in 'mevin benty')
        elif query in disp_low:
            score = 80.0
        # 6. Query in gaia name
        elif gaia_low and query in gaia_low:
            score = 75.0
        # 7. Word-level overlap
        else:
            q_words = set(re.findall(r"\w+", query))
            target_words = set(re.findall(r"\w+", f"{disp_low} {user_low} {gaia_low}"))
            overlap = q_words.intersection(target_words)
            if overlap:
                score = 50.0 + len(overlap) * 10.0

        # Prioritize logged in accounts on score ties
        if p.is_logged_in:
            score += 5.0
        if p.cookies_size > 50000:
            score += 2.0

        if score > best_score:
            best_score = score
            best_profile = p

    if best_profile and best_score >= 50.0:
        return best_profile.binary_cmd, best_profile.profile_dir, best_profile.user_data_dir

    # Fallback to top profile in filtered list
    top = filtered[0]
    return top.binary_cmd, top.profile_dir, top.user_data_dir
