"""Phase 5 — Email engine on the Gmail API.

Backed by google-api-python-client + google-auth-oauthlib (optional install:
``pip install jarvis[gmail]``). First run opens a browser OAuth flow using
``data/gmail-credentials.json`` (Google Cloud Console → OAuth client); the
token is cached at ``data/gmail-token.json``.

Capabilities:
- search / list messages (GREEN)
- read a thread (GREEN)
- classify + summarize recent mail with the LLM: recruiters, interviews,
  application updates (GREEN)
- create a draft (YELLOW — approval required)
- send an email (RED — explicit approval)
"""

from __future__ import annotations

import base64
import logging
import re
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.llm import LLMRouter
from ..db.database import Database, utcnow
from ..profile.profile import Profile
from ..safety.permissions import RiskLevel
from .base import Tool, ToolResult

logger = logging.getLogger("jarvis.email")

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]  # read + draft, includes send
MAX_LIST = 25

CATEGORIES = ["recruiter", "interview", "application_update", "rejection", "other"]


class GmailEngine:
    """Thin, lazy wrapper around the Gmail API service."""

    def __init__(self, credentials_path: Path, token_path: Path) -> None:
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self._service: Any = None

    # ------------------------------------------------------------------ setup
    def available(self) -> bool:
        try:
            import googleapiclient  # noqa: F401
            return True
        except ImportError:
            return False

    def availability_message(self) -> str:
        if not self.available():
            return ("google-api-python-client not installed. "
                    "Run: pip install \"jarvis[gmail]\" "
                    "(google-api-python-client google-auth google-auth-oauthlib)")
        if not self.credentials_path.is_file():
            return (f"OAuth client file missing: {self.credentials_path}. "
                    "Download it from Google Cloud Console → APIs & Services → "
                    "Credentials → OAuth client ID (Desktop app).")
        return "ready"

    def service(self) -> Any:
        if self._service is not None:
            return self._service
        if not self.available():
            raise RuntimeError(self.availability_message())
        if not self.credentials_path.is_file():
            raise RuntimeError(self.availability_message())

        from google.auth.transport.requests import Request  # type: ignore
        from google.oauth2.credentials import Credentials  # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
        from googleapiclient.discovery import build  # type: ignore

        creds: Optional[Any] = None
        if self.token_path.is_file():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds is not None and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if creds is None or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
            logger.info("Gmail token cached at %s", self.token_path)
        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    # ------------------------------------------------------------------- read
    def search(self, query: str, max_results: int = MAX_LIST) -> List[Dict[str, Any]]:
        """Search messages; returns thread summaries with snippets."""
        service = self.service()
        response = service.users().messages().list(
            userId="me", q=query, maxResults=min(max_results, 50)).execute()
        messages = response.get("messages", [])
        threads: List[Dict[str, Any]] = []
        for message_ref in messages:
            message = service.users().messages().get(
                userId="me", id=message_ref["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"]).execute()
            headers = {h["name"]: h["value"] for h in message["payload"]["headers"]}
            threads.append({
                "id": message["id"],
                "thread_id": message.get("threadId", message["id"]),
                "subject": headers.get("Subject", "(no subject)"),
                "from": headers.get("From", ""),
                "date": headers.get("Date", ""),
                "snippet": message.get("snippet", ""),
            })
        return threads

    def read_thread(self, thread_id: str, max_chars: int = 6000) -> Dict[str, Any]:
        """Full text of the latest message in a thread."""
        service = self.service()
        thread = service.users().threads().get(
            userId="me", id=thread_id, format="full").execute()
        messages = thread.get("messages", [])
        if not messages:
            return {"thread_id": thread_id, "text": ""}
        latest = messages[-1]
        headers = {h["name"]: h["value"] for h in latest["payload"]["headers"]}
        text = self._extract_body(latest)
        return {
            "thread_id": thread_id,
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "text": text[:max_chars],
        }

    @staticmethod
    def _extract_body(message: Dict[str, Any]) -> str:
        def walk(part: Dict[str, Any]) -> str:
            mime = part.get("mimeType", "")
            body = part.get("body", {})
            if mime == "text/plain" and body.get("data"):
                return base64.urlsafe_b64decode(body["data"]).decode("utf-8", "replace")
            for sub in part.get("parts", []) or []:
                found = walk(sub)
                if found:
                    return found
            if mime == "text/html" and body.get("data"):
                html = base64.urlsafe_b64decode(body["data"]).decode("utf-8", "replace")
                return re.sub(r"<[^>]+>", " ", html)
            return ""

        return walk(message.get("payload", {})) or message.get("snippet", "")

    # ------------------------------------------------------------------ write
    def create_draft(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        service = self.service()
        raw = MIMEText(body)
        raw["to"] = to
        raw["subject"] = subject
        encoded = base64.urlsafe_b64encode(raw.as_bytes()).decode()
        draft = service.users().drafts().create(
            userId="me", body={"message": {"raw": encoded}}).execute()
        return {"draft_id": draft["id"], "to": to, "subject": subject}

    def send(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        service = self.service()
        raw = MIMEText(body)
        raw["to"] = to
        raw["subject"] = subject
        encoded = base64.urlsafe_b64encode(raw.as_bytes()).decode()
        sent = service.users().messages().send(
            userId="me", body={"raw": encoded}).execute()
        return {"message_id": sent["id"], "to": to, "subject": subject}

    def send_draft(self, draft_id: str) -> Dict[str, Any]:
        service = self.service()
        sent = service.users().drafts().send(
            userId="me", body={"id": draft_id}).execute()
        return {"message_id": sent["id"]}


# ------------------------------------------------------------------ LLM tools
class GmailTools:
    """Bundles GmailEngine operations as permission-aware LLM tools."""

    def __init__(self, engine: GmailEngine, router: LLMRouter,
                 db: Database, profile: Optional[Profile] = None) -> None:
        self.engine = engine
        self.router = router
        self.db = db
        self.profile = profile

    # ---------------------------------------------------------- classification
    def classify_and_summarize(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify a batch of messages with the fast model; persist to DB."""
        if not messages:
            return []
        lines = []
        for index, message in enumerate(messages):
            lines.append(f"[{index}] From: {message['from']}\n"
                         f"Subject: {message['subject']}\n"
                         f"Snippet: {message['snippet'][:200]}")
        prompt = ("Classify each email and write a one-line summary. "
                  f"Categories: {', '.join(CATEGORIES)}.\n"
                  "Return JSON: a list of objects with fields "
                  "index, category, summary, actionable (bool), "
                  "recruiter_email (string or empty).\n\nEmails:\n" + "\n\n".join(lines))
        try:
            parsed = self.router.ask_json(prompt, role="fast")
            results = parsed if isinstance(parsed, list) else parsed.get("results", [])
        except Exception as exc:  # noqa: BLE001 - fall back to keyword heuristics
            logger.warning("LLM classification failed (%s); using heuristics", exc)
            results = []
            for index, message in enumerate(messages):
                text = (message["subject"] + " " + message["snippet"]).lower()
                if re.search(r"interview|schedule a call|technical screen", text):
                    category = "interview"
                elif re.search(r"recruiter|talent|opportunity|role at|position at", text):
                    category = "recruiter"
                elif re.search(r"application|we regret|moved forward with other", text):
                    category = ("rejection" if "regret" in text or "other candidates" in text
                                else "application_update")
                else:
                    category = "other"
                results.append({"index": index, "category": category,
                                "summary": message["snippet"][:120],
                                "actionable": category in ("interview", "recruiter"),
                                "recruiter_email": ""})

        enriched: List[Dict[str, Any]] = []
        for item in results:
            try:
                index = int(item.get("index", -1))
                message = messages[index]
            except (ValueError, IndexError):
                continue
            record = {
                **message,
                "category": str(item.get("category", "other"))[:40],
                "summary": str(item.get("summary", ""))[:300],
                "actionable": bool(item.get("actionable", False)),
            }
            self.db.upsert_email_thread({
                "gmail_thread_id": record["thread_id"],
                "subject": record["subject"],
                "from_email": record["from"],
                "snippet": record["snippet"][:300],
                "category": record["category"],
                "summary": record["summary"],
                "received_at": record.get("date") or utcnow(),
            })
            if record["category"] == "recruiter":
                email_match = re.search(r"<([^>]+)>", record["from"])
                recruiter_email = (item.get("recruiter_email")
                                   or (email_match.group(1) if email_match else record["from"]))
                name_match = re.match(r"^(.*?)\s*<", record["from"])
                self.db.upsert_recruiter(
                    name=name_match.group(1).strip('"') if name_match else "",
                    email=recruiter_email, company="")
            enriched.append(record)
        return enriched

    def daily_digest(self, query: str = "newer_than:7d category:primary",
                     max_results: int = 25) -> str:
        """Fetch, classify and render a digest of recent mail."""
        messages = self.engine.search(query, max_results=max_results)
        if not messages:
            return "No recent messages matched."
        enriched = self.classify_and_summarize(messages)
        counts: Dict[str, int] = {}
        lines: List[str] = []
        for record in enriched:
            counts[record["category"]] = counts.get(record["category"], 0) + 1
            flag = " [ACTION]" if record["actionable"] else ""
            lines.append(f"- ({record['category']}) {record['subject']}{flag} — "
                         f"{record['summary']}")
        header = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        return f"Recent mail — {header}\n" + "\n".join(lines[:30])

    # --------------------------------------------------------------- drafting
    def draft_reply(self, thread_id: str, extra_instructions: str = "") -> Dict[str, Any]:
        """LLM-draft a reply to a thread (saved as a Gmail DRAFT, never sent)."""
        thread = self.engine.read_thread(thread_id)
        signature = f"\n\nBest,\n{self.profile.name}" if self.profile else ""
        prompt = (f"Draft a professional, concise reply to this email.\n"
                  f"Subject: {thread['subject']}\nFrom: {thread['from']}\n\n"
                  f"{thread['text'][:4000]}\n\n"
                  f"Instructions: {extra_instructions or 'keep it brief and polite'}")
        body = self.router.chat(prompt, role="reasoning") + signature
        return self.engine.create_draft(
            to=re.sub(r".*<|>.*", "", thread["from"]) or thread["from"],
            subject="Re: " + thread["subject"],
            body=body)


# -------------------------------------------------------------- tool wrappers
def _unavailable(engine: GmailEngine) -> Optional[str]:
    if not engine.available():
        return engine.availability_message()
    if not engine.credentials_path.is_file():
        return engine.availability_message()
    return None


class EmailSearchTool(Tool):
    name = "email.search"
    description = "Search the user's Gmail with Gmail search syntax (e.g. 'from:recruiter newer_than:7d'). Returns subjects, senders and snippets."
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "query": {"type": "string", "description": "Gmail search query"},
            "max_results": {"type": "integer", "description": "Max results (default 15)"},
        },
        "required": ["query"],
    }

    def __init__(self, engine: GmailEngine, gmail_tools: GmailTools) -> None:
        self.engine = engine
        self.gmail_tools = gmail_tools

    def run(self, query: str = "", max_results: int = 15, **_: Any) -> ToolResult:
        if not query:
            return ToolResult.failure("No query provided")
        problem = _unavailable(self.engine)
        if problem:
            return ToolResult.failure(problem)
        try:
            messages = self.engine.search(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Gmail search failed: {exc}")
        lines = [f"- {m['subject']} — {m['from']} :: {m['snippet'][:100]}"
                 for m in messages]
        return ToolResult.success("\n".join(lines) or "No results.",
                                  count=len(messages), messages=messages)


class EmailDigestTool(Tool):
    name = "email.digest"
    description = ("Fetch recent inbox mail, classify it (recruiter / interview / "
                   "application_update / rejection / other), summarize each and "
                   "persist to the database. Great first step for 'check my email'.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "query": {"type": "string", "description": "Gmail query (default: last 7 days primary)"},
        },
        "required": [],
    }

    def __init__(self, gmail_tools: GmailTools) -> None:
        self.gmail_tools = gmail_tools

    def run(self, query: str = "", **_: Any) -> ToolResult:
        problem = _unavailable(self.gmail_tools.engine)
        if problem:
            return ToolResult.failure(problem)
        try:
            digest = self.gmail_tools.daily_digest(query or "newer_than:7d category:primary")
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Email digest failed: {exc}")
        return ToolResult.success(digest)


class EmailReadThreadTool(Tool):
    name = "email.read_thread"
    description = "Read the full latest message of a Gmail thread by thread id."
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {"thread_id": {"type": "string"}},
        "required": ["thread_id"],
    }

    def __init__(self, engine: GmailEngine) -> None:
        self.engine = engine

    def run(self, thread_id: str = "", **_: Any) -> ToolResult:
        if not thread_id:
            return ToolResult.failure("No thread_id provided")
        problem = _unavailable(self.engine)
        if problem:
            return ToolResult.failure(problem)
        try:
            thread = self.engine.read_thread(thread_id)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Reading thread failed: {exc}")
        return ToolResult.success(
            f"Subject: {thread['subject']}\nFrom: {thread['from']}\n\n{thread['text']}",
            thread_id=thread_id)


class EmailDraftTool(Tool):
    name = "email.create_draft"
    description = ("Create a Gmail DRAFT (never sends). Requires 'to', 'subject', "
                   "'body'. The user reviews it in Gmail before sending.")
    risk = RiskLevel.YELLOW
    parameters = {
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, engine: GmailEngine) -> None:
        self.engine = engine

    def run(self, to: str = "", subject: str = "", body: str = "", **_: Any) -> ToolResult:
        problem = _unavailable(self.engine)
        if problem:
            return ToolResult.failure(problem)
        if not (to and subject and body):
            return ToolResult.failure("Draft needs to, subject and body")
        try:
            draft = self.engine.create_draft(to, subject, body)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Creating draft failed: {exc}")
        return ToolResult.success(
            f"Draft created (id {draft['draft_id']}) to {draft['to']}: "
            f"'{draft['subject']}'. It is NOT sent — review it in Gmail.",
            draft_id=draft["draft_id"])


class EmailSendTool(Tool):
    name = "email.send"
    description = ("SEND an email immediately. Extremely destructive — only use "
                   "when the user explicitly asked to send this exact email.")
    risk = RiskLevel.RED
    parameters = {
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, engine: GmailEngine) -> None:
        self.engine = engine

    def run(self, to: str = "", subject: str = "", body: str = "", **_: Any) -> ToolResult:
        problem = _unavailable(self.engine)
        if problem:
            return ToolResult.failure(problem)
        if not (to and subject and body):
            return ToolResult.failure("Send needs to, subject and body")
        try:
            sent = self.engine.send(to, subject, body)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Sending failed: {exc}")
        return ToolResult.success(f"Email sent to {sent['to']}: '{sent['subject']}'",
                                  message_id=sent["message_id"])


def register_email_tools(registry: Any, engine: GmailEngine,
                         gmail_tools: GmailTools) -> None:
    registry.register(EmailSearchTool(engine, gmail_tools))
    registry.register(EmailDigestTool(gmail_tools))
    registry.register(EmailReadThreadTool(engine))
    registry.register(EmailDraftTool(engine))
    registry.register(EmailSendTool(engine))


__all__ = ["GmailEngine", "GmailTools", "register_email_tools"]
