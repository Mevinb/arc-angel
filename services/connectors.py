"""
Connector framework: OAuth/credential vault, sync, and approval-gated actions.

Real credentials are encrypted in the local vault.  Sync operations are
read-only; any action that sends, publishes or changes an external system must
pass the normal policy engine first.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from xml.etree import ElementTree

import httpx

from memory.database import db_session
from services.audit import AuditService
from services.crypto import (
    base64url, delete_secret, get_secret, redact_obj, redact_text, set_secret,
)
from services.events import NotificationService
from services.policies import Decision, PolicyService
from services.safety import SafetyService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _challenge(verifier: str) -> str:
    return base64url(hashlib.sha256(verifier.encode("utf-8")).digest())


CATALOG: Dict[str, Dict[str, Any]] = {
    "gmail": {
        "name": "Gmail", "category": "Communication", "auth": "oauth",
        "description": "Read recent mail and send approved messages.",
        "scopes": ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.send"],
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "env_prefix": "GOOGLE",
    },
    "google_calendar": {
        "name": "Google Calendar", "category": "Calendar", "auth": "oauth",
        "description": "Read upcoming events and create approved events.",
        "scopes": ["https://www.googleapis.com/auth/calendar.readonly", "https://www.googleapis.com/auth/calendar.events"],
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "env_prefix": "GOOGLE",
    },
    "google_drive": {
        "name": "Google Drive", "category": "Files", "auth": "oauth",
        "description": "Index selected Drive metadata and documents.",
        "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "env_prefix": "GOOGLE",
    },
    "github": {
        "name": "GitHub", "category": "Developer", "auth": "oauth",
        "description": "Sync notifications and repositories; create approved issues.",
        "scopes": ["notifications", "repo"],
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "env_prefix": "GITHUB",
    },
    "slack": {
        "name": "Slack", "category": "Communication", "auth": "oauth",
        "description": "Read channels and send approved messages.",
        "scopes": ["channels:read", "channels:history", "chat:write"],
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "env_prefix": "SLACK",
    },
    "notion": {
        "name": "Notion", "category": "Knowledge", "auth": "oauth",
        "description": "Search pages and append approved notes.",
        "scopes": [],
        "auth_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "env_prefix": "NOTION",
    },
    "linear": {
        "name": "Linear", "category": "Developer", "auth": "oauth",
        "description": "Sync assigned issues and create approved issues.",
        "scopes": ["read", "issues:create"],
        "auth_url": "https://linear.app/oauth/authorize",
        "token_url": "https://api.linear.app/oauth/token",
        "env_prefix": "LINEAR",
    },
    "spotify": {
        "name": "Spotify", "category": "Media", "auth": "oauth",
        "description": "Read playlists and currently playing context.",
        "scopes": ["user-read-private", "user-read-currently-playing"],
        "auth_url": "https://accounts.spotify.com/authorize",
        "token_url": "https://accounts.spotify.com/api/token",
        "env_prefix": "SPOTIFY",
    },
    "rss": {
        "name": "RSS / Atom", "category": "News", "auth": "none",
        "description": "Sync public feeds without credentials.",
        "scopes": [],
    },
    "webhook": {
        "name": "Webhook", "category": "Automation", "auth": "secret",
        "description": "Send approved JSON to an HTTPS endpoint.",
        "scopes": [],
    },
}

SECRET_PREFIX = "connector:"


class ConnectorService:
    # ------------------------------------------------------------------ catalog
    @classmethod
    def catalog(cls) -> List[Dict[str, Any]]:
        out = []
        for cid, meta in CATALOG.items():
            client_id, client_secret = cls._oauth_credentials(cid)
            configured = meta["auth"] in ("none", "secret") or bool(client_id and client_secret)
            out.append({
                "id": cid, **meta, "configured": configured,
                "setup": cls._setup_hint(cid),
            })
        return out

    @classmethod
    def list_connections(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT * FROM connector_connections ORDER BY created_at DESC"
            ).fetchall()
        return [cls._connection_row(r) for r in rows]

    @classmethod
    def get_connection(cls, connection_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM connector_connections WHERE id=?", (connection_id,)).fetchone()
        return cls._connection_row(row) if row else None

    # ----------------------------------------------------------------- OAuth
    @classmethod
    def oauth_url(
        cls, connector_id: str, redirect_uri: str, *, account_label: str = "Primary",
    ) -> Dict[str, Any]:
        meta = CATALOG.get(connector_id)
        if not meta or meta.get("auth") != "oauth":
            raise ValueError("Connector does not use OAuth")
        client_id, client_secret = cls._oauth_credentials(connector_id)
        if not client_id or not client_secret:
            raise ValueError(f"{meta['name']} OAuth is not configured. Set {cls._env_prefix(connector_id)}_CLIENT_ID and _CLIENT_SECRET.")
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        set_secret(f"oauth_state:{state}", {
            "connector_id": connector_id, "redirect_uri": redirect_uri,
            "verifier": verifier, "account_label": account_label,
        })
        params = {
            "client_id": client_id, "redirect_uri": redirect_uri,
            "response_type": "code", "state": state,
            "code_challenge": _challenge(verifier), "code_challenge_method": "S256",
        }
        if meta["scopes"]:
            params["scope"] = " ".join(meta["scopes"])
        if connector_id in ("gmail", "google_calendar", "google_drive"):
            params.update({"access_type": "offline", "prompt": "consent"})
        if connector_id == "notion":
            params["owner"] = "user"
        with db_session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO connector_states (state, connector_id, redirect_uri, created_at) VALUES (?, ?, ?, ?)",
                (state, connector_id, redirect_uri, _now()),
            )
        return {
            "connector_id": connector_id, "state": state,
            "url": f"{meta['auth_url']}?{urlencode(params)}",
            "scopes": meta["scopes"],
        }

    @classmethod
    async def complete_oauth(cls, code: str, state: str) -> Dict[str, Any]:
        state_data = get_secret(f"oauth_state:{state}")
        if not state_data:
            raise ValueError("OAuth state is unknown or expired")
        delete_secret(f"oauth_state:{state}")
        with db_session() as conn:
            row = conn.execute("SELECT * FROM connector_states WHERE state=?", (state,)).fetchone()
            if not row:
                raise ValueError("OAuth state was not issued by this app")
            conn.execute("DELETE FROM connector_states WHERE state=?", (state,))
        connector_id = state_data["connector_id"]
        meta = CATALOG[connector_id]
        client_id, client_secret = cls._oauth_credentials(connector_id)
        data = {
            "client_id": client_id, "client_secret": client_secret, "code": code,
            "redirect_uri": state_data["redirect_uri"], "grant_type": "authorization_code",
            "code_verifier": state_data["verifier"],
        }
        headers = {"Accept": "application/json"}
        if connector_id == "notion":
            headers["Authorization"] = "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(meta["token_url"], data=data, headers=headers)
        try:
            payload = res.json()
        except Exception:
            payload = {"raw": res.text[:1000]}
        if res.status_code >= 400 or payload.get("error"):
            raise ValueError(f"OAuth exchange failed: {payload.get('error_description') or payload.get('error') or res.text[:300]}")
        access_token = payload.get("access_token") or payload.get("authed_user", {}).get("access_token")
        if not access_token:
            raise ValueError("OAuth provider did not return an access token")
        refresh_token = payload.get("refresh_token")
        expires_at = None
        if payload.get("expires_in"):
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(payload["expires_in"]))).isoformat()
        return cls._create_connection(
            connector_id, state_data.get("account_label") or "Primary",
            access_token=access_token, refresh_token=refresh_token,
            expires_at=expires_at, scopes=meta["scopes"], metadata=payload,
        )

    # ---------------------------------------------------------- credential mode
    @classmethod
    def connect_credentials(
        cls, connector_id: str, *, account_label: str = "Primary",
        credentials: Optional[Dict[str, Any]] = None, scopes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = CATALOG.get(connector_id)
        if not meta:
            raise ValueError("Unknown connector")
        credentials = credentials or {}
        if meta["auth"] == "oauth" and not credentials.get("access_token"):
            raise ValueError("Provide an access_token, or use the OAuth flow")
        if connector_id == "rss" and not (credentials.get("feed_url") or (metadata or {}).get("feed_url")):
            raise ValueError("RSS connectors require feed_url")
        if connector_id == "webhook" and not (credentials.get("url") or (metadata or {}).get("url")):
            raise ValueError("Webhook connectors require url")
        public_meta = {
            key: value for key, value in credentials.items()
            if key in ("feed_url", "url", "base_url", "username", "workspace")
        }
        return cls._create_connection(
            connector_id, account_label,
            access_token=credentials.get("access_token"),
            refresh_token=credentials.get("refresh_token"),
            expires_at=credentials.get("expires_at"),
            scopes=scopes or meta["scopes"],
            metadata={**(metadata or {}), **public_meta},
        )

    @classmethod
    def _create_connection(
        cls, connector_id: str, account_label: str, *, access_token: Optional[str],
        refresh_token: Optional[str] = None, expires_at: Optional[str] = None,
        scopes: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        connection_id = f"cx_{uuid.uuid4().hex[:12]}"
        secret_name = f"{SECRET_PREFIX}{connector_id}:{connection_id}"
        set_secret(secret_name, {
            "access_token": access_token, "refresh_token": refresh_token,
            "expires_at": expires_at, **(metadata or {}),
        })
        now = _now()
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO connector_connections (
                    id, connector_id, account_label, status, scopes, metadata,
                    access_token_secret, refresh_token_secret, expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, 'connected', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    connection_id, connector_id, account_label[:120], json.dumps(scopes or []),
                    json.dumps({k: v for k, v in (metadata or {}).items() if k != "credentials"}),
                    secret_name, secret_name, expires_at, now, now,
                ),
            )
        AuditService.log(
            "connector", "connector.connected", actor="user", connector_id=connector_id,
            summary=f"Connected {CATALOG[connector_id]['name']} ({account_label})",
            data={"connection_id": connection_id, "scopes": scopes or []},
            activity_title=f"Connected {CATALOG[connector_id]['name']}",
        )
        return cls.get_connection(connection_id)  # type: ignore[return-value]

    @classmethod
    def disconnect(cls, connection_id: str) -> bool:
        connection = cls.get_connection(connection_id)
        if not connection:
            return False
        delete_secret(connection["access_token_secret"])
        with db_session() as conn:
            conn.execute("DELETE FROM connector_items WHERE connection_id=?", (connection_id,))
            conn.execute("DELETE FROM connector_connections WHERE id=?", (connection_id,))
        AuditService.log(
            "connector", "connector.disconnected", actor="user",
            connector_id=connection["connector_id"],
            summary=f"Disconnected {connection['connector_id']} ({connection['account_label']})",
        )
        return True

    @classmethod
    async def test(cls, connection_id: str) -> Dict[str, Any]:
        connection = cls.get_connection(connection_id)
        if not connection:
            raise ValueError("Connection not found")
        connector_id = connection["connector_id"]
        try:
            if connector_id == "rss":
                result = await cls._fetch_rss(connection)
                status, detail = "ok", f"Feed returned {len(result)} items"
            elif connector_id == "webhook":
                status, detail = "ok", "Webhook target is configured"
            else:
                token = await cls._access_token(connection)
                headers = cls._auth_headers(connector_id, token)
                endpoint, params = cls._probe(connector_id)
                async with httpx.AsyncClient(timeout=20.0) as client:
                    res = await client.get(endpoint, headers=headers, params=params)
                if res.status_code >= 400:
                    raise ValueError(f"HTTP {res.status_code}: {res.text[:200]}")
                status, detail = "ok", f"HTTP {res.status_code}"
            with db_session() as conn:
                conn.execute(
                    "UPDATE connector_connections SET status='connected', error=NULL, updated_at=? WHERE id=?",
                    (_now(), connection_id),
                )
            return {"ok": True, "status": status, "detail": detail}
        except Exception as exc:
            with db_session() as conn:
                conn.execute(
                    "UPDATE connector_connections SET status='error', error=?, updated_at=? WHERE id=?",
                    (redact_text(str(exc))[:1000], _now(), connection_id),
                )
            return {"ok": False, "error": str(exc)}

    @classmethod
    async def sync(cls, connection_id: str, limit: int = 25) -> Dict[str, Any]:
        if SafetyService.offline_mode():
            raise ValueError("Offline mode is enabled; connector sync is unavailable")
        connection = cls.get_connection(connection_id)
        if not connection:
            raise ValueError("Connection not found")
        connector_id = connection["connector_id"]
        try:
            items = await cls._sync_dispatch(connection, max(1, min(limit, 100)))
            saved = 0
            with db_session() as conn:
                for item in items:
                    content = item.get("content") or item.get("body") or ""
                    digest = hashlib.sha256((item.get("external_id", "") + content).encode()).hexdigest()
                    exists = conn.execute(
                        "SELECT id FROM connector_items WHERE connection_id=? AND content_hash=?",
                        (connection_id, digest),
                    ).fetchone()
                    if exists:
                        continue
                    conn.execute(
                        """
                        INSERT INTO connector_items (
                            connection_id, connector_id, external_id, item_type, title, body,
                            author, url, occurred_at, content_hash, metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            connection_id, connector_id, str(item.get("external_id", ""))[:300],
                            item.get("item_type", "item")[:60], redact_text(item.get("title", ""))[:300],
                            content[:50000], item.get("author"), item.get("url"),
                            item.get("occurred_at"), digest, json.dumps(item.get("metadata") or {}), _now(),
                        ),
                    )
                    saved += 1
                conn.execute(
                    "UPDATE connector_connections SET last_sync_at=?, status='connected', error=NULL, updated_at=? WHERE id=?",
                    (_now(), _now(), connection_id),
                )
            AuditService.log(
                "connector", "connector.synced", actor="system", connector_id=connector_id,
                summary=f"Synced {saved} new items from {connector_id}",
                data={"connection_id": connection_id, "fetched": len(items), "saved": saved},
                activity_title=f"Synced {CATALOG.get(connector_id, {}).get('name', connector_id)}",
            )
            return {"ok": True, "fetched": len(items), "saved": saved}
        except Exception as exc:
            with db_session() as conn:
                conn.execute(
                    "UPDATE connector_connections SET status='error', error=?, updated_at=? WHERE id=?",
                    (redact_text(str(exc))[:1000], _now(), connection_id),
                )
            raise

    @classmethod
    def items(cls, connection_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        where, params = "", []
        if connection_id:
            where = "WHERE connection_id=?"
            params.append(connection_id)
        params.append(max(1, min(limit, 500)))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM connector_items {where} ORDER BY COALESCE(occurred_at, created_at) DESC LIMIT ?",
                params,
            ).fetchall()
        out = []
        for row in rows:
            try:
                meta = json.loads(row["metadata"] or "{}")
            except Exception:
                meta = {}
            out.append({
                "id": row["id"], "connection_id": row["connection_id"], "connector_id": row["connector_id"],
                "external_id": row["external_id"], "item_type": row["item_type"],
                "title": row["title"], "body": row["body"], "author": row["author"],
                "url": row["url"], "occurred_at": row["occurred_at"],
                "metadata": meta, "created_at": row["created_at"],
            })
        return out

    # --------------------------------------------------------------- actions
    @classmethod
    def prepare_action(
        cls, connection_id: str, action: str, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        connection = cls.get_connection(connection_id)
        if not connection:
            raise ValueError("Connection not found")
        risk = "external_communication" if action in ("send_email", "send_message", "post_webhook") else "reversible_write"
        domain = None
        if payload.get("url"):
            from urllib.parse import urlparse

            domain = urlparse(payload["url"]).hostname
        context = {
            "action": f"connector:{action}", "tool_name": f"connector.{action}",
            "risk_class": risk, "connector_id": connection["connector_id"],
            "domain": domain, "contact": payload.get("to") or payload.get("channel"),
            "interactive": True, "autopilot": False,
        }
        decision = PolicyService.evaluate(context)
        if decision["decision"] == Decision.DENY:
            return {"status": "denied", "reason": "; ".join(decision.get("reasons") or [])}
        if decision["decision"] == Decision.REQUIRE_APPROVAL:
            approval = PolicyService.request_approval(
                context, reason="; ".join(decision.get("reasons") or ["Approval required"]),
                payload={"connection_id": connection_id, "action": action, "payload": redact_obj(payload)},
            )
            return {"status": "awaiting_approval", "approval_id": approval["id"]}
        return {"status": "ready", "decision": decision}

    @classmethod
    async def execute_action(
        cls, connection_id: str, action: str, payload: Dict[str, Any],
        *, approval_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        connection = cls.get_connection(connection_id)
        if not connection:
            raise ValueError("Connection not found")
        if approval_id:
            approval = PolicyService.get_approval(approval_id)
            if not approval or approval["status"] != "approved":
                raise PermissionError("Approval is not valid")
            approved_payload = (approval.get("payload") or {})
            if approved_payload.get("connection_id") != connection_id or approved_payload.get("action") != action:
                raise PermissionError("Approval does not match this connector action")
            if json.dumps(approved_payload.get("payload") or {}, sort_keys=True, default=str) != json.dumps(payload, sort_keys=True, default=str):
                raise PermissionError("Approval payload does not match this action")
        connector_id = connection["connector_id"]
        if action in ("send_email", "send_message", "post_webhook", "create_issue", "create_event"):
            risk = "external_communication" if action in ("send_email", "send_message", "post_webhook") else "reversible_write"
            decision = PolicyService.evaluate({
                "action": f"connector:{action}", "tool_name": f"connector.{action}",
                "risk_class": risk, "connector_id": connector_id,
                "contact": payload.get("to") or payload.get("channel"),
                "interactive": True, "autopilot": False,
            })
            if decision["decision"] == Decision.DENY:
                raise PermissionError("; ".join(decision.get("reasons") or ["Denied by policy"]))
            if decision["decision"] == Decision.REQUIRE_APPROVAL and not approval_id:
                raise PermissionError("Approval required for this external action")
        result = await cls._execute_dispatch(connection, action, payload)
        AuditService.log(
            "connector", f"connector.{action}", actor="user", connector_id=connector_id,
            approval_id=approval_id, status="ok",
            summary=f"{CATALOG.get(connector_id, {}).get('name', connector_id)}: {action}",
            data={"connection_id": connection_id, "payload": redact_obj(payload), "result": redact_obj(result)},
            activity_title=f"Connector action: {action}",
        )
        return {"ok": True, "action": action, "result": result}

    # --------------------------------------------------------------- internals
    @classmethod
    async def _sync_dispatch(cls, connection: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        connector_id = connection["connector_id"]
        if connector_id == "rss":
            return await cls._fetch_rss(connection)
        token = await cls._access_token(connection)
        headers = cls._auth_headers(connector_id, token)
        if connector_id == "github":
            async with httpx.AsyncClient(timeout=30.0) as client:
                notif = await client.get("https://api.github.com/notifications", headers=headers, params={"per_page": limit})
                repos = await client.get("https://api.github.com/user/repos", headers=headers, params={"sort": "updated", "per_page": limit})
            items = []
            if notif.status_code < 400:
                for n in notif.json():
                    items.append({
                        "external_id": n.get("id"), "item_type": "notification",
                        "title": n.get("subject", {}).get("title", ""),
                        "content": n.get("subject", {}).get("type", ""), "url": n.get("subject", {}).get("url"),
                        "occurred_at": n.get("updated_at"), "metadata": {"repo": n.get("repository", {}).get("full_name")},
                    })
            if repos.status_code < 400:
                for r in repos.json():
                    items.append({
                        "external_id": r.get("id"), "item_type": "repository",
                        "title": r.get("full_name", ""), "content": r.get("description") or "",
                        "url": r.get("html_url"), "occurred_at": r.get("updated_at"),
                    })
            return items[:limit]
        if connector_id == "gmail":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get("https://gmail.googleapis.com/gmail/v1/users/me/messages", headers=headers, params={"maxResults": limit})
                if res.status_code >= 400:
                    raise ValueError(f"Gmail HTTP {res.status_code}: {res.text[:200]}")
                items = []
                for msg in res.json().get("messages", []):
                    detail = await client.get(
                        f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}",
                        headers=headers, params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
                    )
                    data = detail.json() if detail.status_code < 400 else {}
                    headers_list = data.get("payload", {}).get("headers", [])
                    meta = {h.get("name"): h.get("value") for h in headers_list}
                    items.append({
                        "external_id": msg["id"], "item_type": "email", "title": meta.get("Subject", "(no subject)"),
                        "content": data.get("snippet", ""), "author": meta.get("From"),
                        "occurred_at": meta.get("Date"), "metadata": {"threadId": data.get("threadId")},
                    })
            return items
        if connector_id == "google_calendar":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get(
                    "https://www.googleapis.com/calendar/v3/calendars/primary/events", headers=headers,
                    params={"maxResults": limit, "singleEvents": "true", "orderBy": "startTime",
                            "timeMin": datetime.now(timezone.utc).isoformat()},
                )
            if res.status_code >= 400:
                raise ValueError(f"Calendar HTTP {res.status_code}: {res.text[:200]}")
            return [{
                "external_id": e.get("id"), "item_type": "event", "title": e.get("summary", ""),
                "content": e.get("description", ""), "occurred_at": (e.get("start") or {}).get("dateTime") or (e.get("start") or {}).get("date"),
                "url": e.get("htmlLink"), "metadata": {"location": e.get("location")},
            } for e in res.json().get("items", [])]
        if connector_id == "google_drive":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get(
                    "https://www.googleapis.com/drive/v3/files", headers=headers,
                    params={"pageSize": limit, "orderBy": "modifiedTime desc",
                            "fields": "files(id,name,mimeType,modifiedTime,webViewLink,description)"},
                )
            if res.status_code >= 400:
                raise ValueError(f"Drive HTTP {res.status_code}: {res.text[:200]}")
            return [{
                "external_id": f.get("id"), "item_type": "drive_file", "title": f.get("name", ""),
                "content": f.get("description", ""), "occurred_at": f.get("modifiedTime"),
                "url": f.get("webViewLink"), "metadata": {"mimeType": f.get("mimeType")},
            } for f in res.json().get("files", [])]
        if connector_id == "slack":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get("https://slack.com/api/conversations.list", headers=headers, params={"limit": limit})
            data = res.json()
            if not data.get("ok"):
                raise ValueError(f"Slack: {data.get('error')}")
            return [{
                "external_id": c.get("id"), "item_type": "channel", "title": c.get("name", ""),
                "content": c.get("topic", {}).get("value", ""), "metadata": {"is_private": c.get("is_private")},
            } for c in data.get("channels", [])]
        if connector_id == "notion":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post("https://api.notion.com/v1/search", headers={**headers, "Notion-Version": "2022-06-28"}, json={"page_size": limit})
            if res.status_code >= 400:
                raise ValueError(f"Notion HTTP {res.status_code}: {res.text[:200]}")
            return [{
                "external_id": r.get("id"), "item_type": "notion_page",
                "title": cls._notion_title(r), "content": "", "occurred_at": r.get("last_edited_time"),
                "url": r.get("url"),
            } for r in res.json().get("results", [])]
        if connector_id == "linear":
            query = "query { viewer { assignedIssues(first: %d) { nodes { id identifier title description url updatedAt } } } }" % limit
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post("https://api.linear.app/graphql", headers=headers, json={"query": query})
            data = res.json()
            nodes = (((data.get("data") or {}).get("viewer") or {}).get("assignedIssues") or {}).get("nodes", [])
            return [{
                "external_id": n.get("id"), "item_type": "issue", "title": n.get("title", ""),
                "content": n.get("description", ""), "url": n.get("url"), "occurred_at": n.get("updatedAt"),
                "metadata": {"identifier": n.get("identifier")},
            } for n in nodes]
        if connector_id == "spotify":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get("https://api.spotify.com/v1/me/player/currently-playing", headers=headers)
            if res.status_code == 204:
                return []
            data = res.json()
            item = data.get("item") or {}
            return [{
                "external_id": item.get("id"), "item_type": "track", "title": item.get("name", ""),
                "content": ", ".join(a.get("name", "") for a in item.get("artists", [])),
                "url": (item.get("external_urls") or {}).get("spotify"),
            }] if item else []
        raise ValueError(f"Sync is not implemented for {connector_id}")

    @classmethod
    async def _fetch_rss(cls, connection: Dict[str, Any]) -> List[Dict[str, Any]]:
        secret = get_secret(connection["access_token_secret"]) or {}
        meta = connection.get("metadata") or {}
        url = secret.get("feed_url") or meta.get("feed_url") or connection.get("url")
        if not url:
            raise ValueError("RSS feed_url is missing")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            res = await client.get(url)
            res.raise_for_status()
        root = ElementTree.fromstring(res.content)
        items = []
        for node in root.iter():
            if node.tag.split("}")[-1] not in ("item", "entry"):
                continue
            values = {}
            for child in node:
                key = child.tag.split("}")[-1]
                if key == "link":
                    values[key] = child.attrib.get("href") or (child.text or "")
                else:
                    values[key] = (child.text or "").strip()
            items.append({
                "external_id": values.get("guid") or values.get("id") or values.get("link", ""),
                "item_type": "feed_item", "title": values.get("title", ""),
                "content": values.get("description") or values.get("summary") or values.get("content", ""),
                "author": values.get("author"), "url": values.get("link"),
                "occurred_at": values.get("pubDate") or values.get("updated") or values.get("published"),
            })
        return items

    @classmethod
    async def _execute_dispatch(cls, connection: Dict[str, Any], action: str, payload: Dict[str, Any]) -> Any:
        connector_id = connection["connector_id"]
        token = await cls._access_token(connection)
        headers = cls._auth_headers(connector_id, token)
        if action == "send_email" and connector_id == "gmail":
            msg = MIMEText(payload.get("body", ""), "plain", "utf-8")
            msg["To"] = payload.get("to", "")
            msg["Subject"] = payload.get("subject", "")
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                    headers=headers, json={"raw": raw},
                )
            if res.status_code >= 400:
                raise ValueError(f"Gmail HTTP {res.status_code}: {res.text[:200]}")
            return {"message_id": res.json().get("id")}
        if action == "send_message" and connector_id == "slack":
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    "https://slack.com/api/chat.postMessage", headers=headers,
                    json={"channel": payload.get("channel"), "text": payload.get("text", "")},
                )
            data = res.json()
            if not data.get("ok"):
                raise ValueError(f"Slack: {data.get('error')}")
            return {"ts": data.get("ts")}
        if action == "post_webhook":
            secret = get_secret(connection["access_token_secret"]) or {}
            url = payload.get("url") or secret.get("url") or connection.get("metadata", {}).get("url")
            if not url:
                raise ValueError("Webhook URL is missing")
            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(url, json=payload.get("json") or payload, headers={"User-Agent": "GROK-Connectors/2"})
            return {"status_code": res.status_code, "body": res.text[:1000]}
        if action == "create_issue" and connector_id == "github":
            repo = payload.get("repo")
            if not repo:
                raise ValueError("repo is required")
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    f"https://api.github.com/repos/{repo}/issues", headers=headers,
                    json={"title": payload.get("title"), "body": payload.get("body", "")},
                )
            if res.status_code >= 400:
                raise ValueError(f"GitHub HTTP {res.status_code}: {res.text[:200]}")
            return {"url": res.json().get("html_url"), "number": res.json().get("number")}
        if action == "create_event" and connector_id == "google_calendar":
            event = {
                "summary": payload.get("title", "Untitled"),
                "description": payload.get("description", ""),
                "start": {"dateTime": payload.get("start")},
                "end": {"dateTime": payload.get("end")},
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(
                    "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                    headers=headers, json=event,
                )
            if res.status_code >= 400:
                raise ValueError(f"Calendar HTTP {res.status_code}: {res.text[:200]}")
            return {"event_id": res.json().get("id"), "url": res.json().get("htmlLink")}
        raise ValueError(f"Unsupported action {action} for {connector_id}")

    @classmethod
    async def _access_token(cls, connection: Dict[str, Any]) -> str:
        secret = get_secret(connection["access_token_secret"]) or {}
        token = secret.get("access_token")
        if not token:
            raise ValueError("Connector has no access token")
        expires = secret.get("expires_at")
        if expires and secret.get("refresh_token"):
            try:
                exp = datetime.fromisoformat(expires)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp <= datetime.now(timezone.utc) + timedelta(seconds=60):
                    return await cls._refresh(connection, secret)
            except Exception:
                pass
        return token

    @classmethod
    async def _refresh(cls, connection: Dict[str, Any], secret: Dict[str, Any]) -> str:
        connector_id = connection["connector_id"]
        meta = CATALOG[connector_id]
        client_id, client_secret = cls._oauth_credentials(connector_id)
        data = {
            "client_id": client_id, "client_secret": client_secret,
            "grant_type": "refresh_token", "refresh_token": secret["refresh_token"],
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(meta["token_url"], data=data, headers={"Accept": "application/json"})
        payload = res.json()
        if res.status_code >= 400 or payload.get("error"):
            raise ValueError(f"Token refresh failed: {payload.get('error_description') or payload.get('error')}")
        secret["access_token"] = payload.get("access_token")
        if payload.get("expires_in"):
            secret["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=int(payload["expires_in"]))).isoformat()
        set_secret(connection["access_token_secret"], secret)
        with db_session() as conn:
            conn.execute(
                "UPDATE connector_connections SET expires_at=?, updated_at=? WHERE id=?",
                (secret.get("expires_at"), _now(), connection["id"]),
            )
        return secret["access_token"]

    @classmethod
    def _auth_headers(cls, connector_id: str, token: str) -> Dict[str, str]:
        if connector_id == "notion":
            return {"Authorization": f"Bearer {token}"}
        if connector_id == "linear":
            return {"Authorization": token}
        return {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": "GROK-Local/2"}

    @classmethod
    def _probe(cls, connector_id: str) -> tuple:
        probes = {
            "github": ("https://api.github.com/user", {}),
            "gmail": ("https://gmail.googleapis.com/gmail/v1/users/me/profile", {}),
            "google_calendar": ("https://www.googleapis.com/calendar/v3/users/me/calendarList", {"maxResults": 1}),
            "google_drive": ("https://www.googleapis.com/drive/v3/about", {"fields": "user"}),
            "slack": ("https://slack.com/api/auth.test", {}),
            "notion": ("https://api.notion.com/v1/users/me", {}),
            "linear": ("https://api.linear.app/graphql", {}),
            "spotify": ("https://api.spotify.com/v1/me", {}),
        }
        return probes.get(connector_id, (f"https://api.{connector_id}.com", {}))

    @classmethod
    def _oauth_credentials(cls, connector_id: str) -> tuple:
        prefix = cls._env_prefix(connector_id)
        return (
            os.getenv(f"GROK_{prefix}_CLIENT_ID", "").strip(),
            os.getenv(f"GROK_{prefix}_CLIENT_SECRET", "").strip(),
        )

    @classmethod
    def _env_prefix(cls, connector_id: str) -> str:
        return CATALOG.get(connector_id, {}).get("env_prefix") or connector_id.upper()

    @classmethod
    def _setup_hint(cls, connector_id: str) -> str:
        if connector_id == "rss":
            return "Add any public RSS/Atom feed URL."
        if connector_id == "webhook":
            return "Add an HTTPS endpoint that accepts JSON."
        prefix = cls._env_prefix(connector_id)
        return f"Set GROK_{prefix}_CLIENT_ID and GROK_{prefix}_CLIENT_SECRET, then use Connect."

    @classmethod
    def _connection_row(cls, row) -> Dict[str, Any]:
        try:
            scopes = json.loads(row["scopes"] or "[]")
        except Exception:
            scopes = []
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}
        return {
            "id": row["id"], "connector_id": row["connector_id"],
            "account_label": row["account_label"], "status": row["status"],
            "scopes": scopes, "metadata": metadata,
            "access_token_secret": row["access_token_secret"],
            "expires_at": row["expires_at"], "last_sync_at": row["last_sync_at"],
            "error": row["error"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @classmethod
    def _notion_title(cls, result: Dict[str, Any]) -> str:
        props = result.get("properties") or {}
        for value in props.values():
            if value.get("type") == "title":
                return "".join(t.get("plain_text", "") for t in value.get("title", [])) or "Untitled"
        title = result.get("title") or []
        return "".join(t.get("plain_text", "") for t in title) or "Untitled"
