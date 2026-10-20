"""
Policy engine, risk classes and approval workflow.

Policy decisions are deterministic and auditable.  Safe defaults:
- read-only actions are allowed;
- writes inside the managed sandbox are allowed by a seeded policy;
- deletion, shell/credential operations, external communication and financial
  actions require explicit approval unless a narrow, explicitly pre-authorized
  policy matches;
- autopilot never executes anything without a matching allow policy;
- the global kill switch denies every action.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR, WORKSPACES_DIR
from memory.database import db_session
from services.crypto import redact_obj, redact_text


class RiskClass:
    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    DESTRUCTIVE_WRITE = "destructive_write"
    EXTERNAL_COMMUNICATION = "external_communication"
    FINANCIAL_IDENTITY = "financial_identity"
    RESTRICTED = "restricted"

    ALL = (
        READ_ONLY, REVERSIBLE_WRITE, DESTRUCTIVE_WRITE,
        EXTERNAL_COMMUNICATION, FINANCIAL_IDENTITY, RESTRICTED,
    )
    # These never execute on autopilot without an explicit pre-authorization.
    HIGH_RISK = (
        DESTRUCTIVE_WRITE, EXTERNAL_COMMUNICATION,
        FINANCIAL_IDENTITY, RESTRICTED,
    )


class Decision:
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


SCOPE_TYPES = ("global", "agent", "tool", "connector", "folder", "domain", "contact", "action", "risk")
EFFECTS = (Decision.ALLOW, Decision.DENY, Decision.REQUIRE_APPROVAL)
SANDBOX_DIR = DATA_DIR / "sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any, default):
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def args_hash(args: Any) -> str:
    """Stable digest binding an approval to the exact tool arguments."""
    canonical = json.dumps(args if args is not None else {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _domain_matches(domain: str, patterns: List[str]) -> bool:
    domain = (domain or "").strip().lower()
    for pattern in patterns:
        p = str(pattern).strip().lower()
        if not p:
            continue
        if p.startswith("*."):
            if domain == p[2:] or domain.endswith(p[1:]):
                return True
        elif fnmatch.fnmatch(domain, p):
            return True
    return False


def _path_matches(path: str, patterns: List[str]) -> bool:
    if not path:
        return False
    try:
        candidate = Path(path).expanduser().resolve()
    except Exception:
        return False
    for pattern in patterns:
        try:
            root = Path(str(pattern)).expanduser().resolve()
            if candidate == root or root in candidate.parents:
                return True
        except Exception:
            continue
    return False


def _in_sandbox(path: str) -> bool:
    return _path_matches(path, [str(WORKSPACES_DIR), str(SANDBOX_DIR)])


class PolicyService:
    SEED_MARKER = "policies_seeded_v1"

    @classmethod
    def ensure_defaults(cls) -> None:
        with db_session() as conn:
            marker = conn.execute(
                "SELECT value FROM platform_meta WHERE key = ?", (cls.SEED_MARKER,)
            ).fetchone()
            if marker:
                return
            now = _now()
            defaults = [
                {
                    "id": "default-read-only",
                    "name": "Read-only tools",
                    "description": "Search, fetch, inspect files and read knowledge without approval.",
                    "priority": 100,
                    "effect": Decision.ALLOW,
                    "scope_type": "risk",
                    "scope_value": RiskClass.READ_ONLY,
                    "risk_classes": json.dumps([RiskClass.READ_ONLY]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 1,
                },
                {
                    "id": "default-sandbox-writes",
                    "name": "Sandbox workspace writes",
                    "description": "Agents may write artifacts inside managed task workspaces.",
                    "priority": 85,
                    "effect": Decision.ALLOW,
                    "scope_type": "folder",
                    "scope_value": str(WORKSPACES_DIR),
                    "risk_classes": json.dumps([RiskClass.REVERSIBLE_WRITE]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 1,
                },
                {
                    "id": "default-sandbox-writes-2",
                    "name": "Scratch sandbox writes",
                    "description": "Agents may write inside the scratch sandbox.",
                    "priority": 84,
                    "effect": Decision.ALLOW,
                    "scope_type": "folder",
                    "scope_value": str(SANDBOX_DIR),
                    "risk_classes": json.dumps([RiskClass.REVERSIBLE_WRITE]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 1,
                },
                {
                    "id": "default-local-memory",
                    "name": "Local memory writes",
                    "description": "Remember/forget operations on the local memory vault.",
                    "priority": 80,
                    "effect": Decision.ALLOW,
                    "scope_type": "tool",
                    "scope_value": "memory.*",
                    "risk_classes": json.dumps([RiskClass.REVERSIBLE_WRITE]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 1,
                },
                {
                    "id": "default-knowledge-writes",
                    "name": "Knowledge vault writes",
                    "description": "Research agents may save notes and sources into the local vault.",
                    "priority": 80,
                    "effect": Decision.ALLOW,
                    "scope_type": "tool",
                    "scope_value": "knowledge.*",
                    "risk_classes": json.dumps([RiskClass.REVERSIBLE_WRITE]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 1,
                },
                {
                    "id": "default-local-notifications",
                    "name": "Local notifications",
                    "description": "Create local notifications; no external delivery.",
                    "priority": 80,
                    "effect": Decision.ALLOW,
                    "scope_type": "tool",
                    "scope_value": "notifications.send",
                    "risk_classes": json.dumps([RiskClass.REVERSIBLE_WRITE]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 1,
                },
                {
                    "id": "default-deletion-approval",
                    "name": "Irreversible deletion approval",
                    "description": "Deleting files or data always needs explicit approval.",
                    "priority": 95,
                    "effect": Decision.REQUIRE_APPROVAL,
                    "scope_type": "risk",
                    "scope_value": RiskClass.DESTRUCTIVE_WRITE,
                    "risk_classes": json.dumps([RiskClass.DESTRUCTIVE_WRITE]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 0,
                },
                {
                    "id": "default-external-communication",
                    "name": "External communication approval",
                    "description": "Email, messages, publishing and webhooks require approval by default.",
                    "priority": 95,
                    "effect": Decision.REQUIRE_APPROVAL,
                    "scope_type": "risk",
                    "scope_value": RiskClass.EXTERNAL_COMMUNICATION,
                    "risk_classes": json.dumps([RiskClass.EXTERNAL_COMMUNICATION]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 0,
                },
                {
                    "id": "default-financial-approval",
                    "name": "Financial and identity approval",
                    "description": "Purchases, transfers and identity-sensitive actions require approval.",
                    "priority": 95,
                    "effect": Decision.REQUIRE_APPROVAL,
                    "scope_type": "risk",
                    "scope_value": RiskClass.FINANCIAL_IDENTITY,
                    "risk_classes": json.dumps([RiskClass.FINANCIAL_IDENTITY]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 0,
                },
                {
                    "id": "default-restricted-approval",
                    "name": "Shell and credential approval",
                    "description": "Shell, desktop and credential operations require approval by default.",
                    "priority": 94,
                    "effect": Decision.REQUIRE_APPROVAL,
                    "scope_type": "risk",
                    "scope_value": RiskClass.RESTRICTED,
                    "risk_classes": json.dumps([RiskClass.RESTRICTED]),
                    "conditions": json.dumps({}),
                    "explicit_allow": 0,
                },
            ]
            for p in defaults:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO policies (
                        id, name, description, enabled, priority, effect, scope_type,
                        scope_value, risk_classes, conditions, explicit_allow,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p["id"], p["name"], p["description"], p["priority"], p["effect"],
                        p["scope_type"], p["scope_value"], p["risk_classes"],
                        p["conditions"], p["explicit_allow"], now, now,
                    ),
                )
            conn.execute(
                "INSERT OR REPLACE INTO platform_meta (key, value, updated_at) VALUES (?, '1', ?)",
                (cls.SEED_MARKER, now),
            )

    # ------------------------------------------------------------------ CRUD
    @classmethod
    def list_policies(cls, enabled_only: bool = False) -> List[Dict[str, Any]]:
        where = "WHERE enabled = 1" if enabled_only else ""
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM policies {where} ORDER BY priority DESC, id ASC"
            ).fetchall()
        return [cls._row(row) for row in rows]

    @classmethod
    def get(cls, policy_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM policies WHERE id = ?", (policy_id,)).fetchone()
        return cls._row(row) if row else None

    @classmethod
    def create(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        cls._validate(payload)
        policy_id = payload.get("id") or f"pol_{uuid.uuid4().hex[:12]}"
        now = _now()
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO policies (
                    id, name, description, enabled, priority, effect, scope_type,
                    scope_value, risk_classes, conditions, rate_limit_per_hour,
                    spend_limit_usd, time_limit_seconds, allow_domains,
                    allow_contacts, allow_folders, explicit_allow, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                cls._insert_values(policy_id, payload, now),
            )
        try:
            from services.audit import AuditService

            AuditService.log(
                "policy", "policy.created", actor="user", summary=f"Created policy {payload.get('name')}",
                data={"policy_id": policy_id, "effect": payload.get("effect")},
            )
        except Exception:
            pass
        return cls.get(policy_id)  # type: ignore[return-value]

    @classmethod
    def update(cls, policy_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = cls.get(policy_id)
        if not current:
            return None
        merged = {**current, **patch, "id": policy_id}
        cls._validate(merged)
        with db_session() as conn:
            conn.execute(
                """
                UPDATE policies SET name=?, description=?, enabled=?, priority=?, effect=?,
                    scope_type=?, scope_value=?, risk_classes=?, conditions=?,
                    rate_limit_per_hour=?, spend_limit_usd=?, time_limit_seconds=?,
                    allow_domains=?, allow_contacts=?, allow_folders=?, explicit_allow=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    merged["name"], merged.get("description", ""), int(bool(merged.get("enabled", True))),
                    int(merged.get("priority", 50)), merged["effect"], merged["scope_type"],
                    merged.get("scope_value"), json.dumps(merged.get("risk_classes") or []),
                    json.dumps(merged.get("conditions") or {}),
                    merged.get("rate_limit_per_hour"), merged.get("spend_limit_usd"),
                    merged.get("time_limit_seconds"), json.dumps(merged.get("allow_domains") or []),
                    json.dumps(merged.get("allow_contacts") or []),
                    json.dumps(merged.get("allow_folders") or []),
                    int(bool(merged.get("explicit_allow"))), _now(), policy_id,
                ),
            )
        try:
            from services.audit import AuditService

            AuditService.log(
                "policy", "policy.updated", actor="user", summary=f"Updated policy {merged.get('name')}",
                data={"policy_id": policy_id, "patch": patch},
            )
        except Exception:
            pass
        return cls.get(policy_id)

    @classmethod
    def delete(cls, policy_id: str) -> bool:
        if policy_id.startswith("default-"):
            raise ValueError("Built-in safety policies cannot be deleted; disable or override them instead.")
        with db_session() as conn:
            cur = conn.execute("DELETE FROM policies WHERE id = ?", (policy_id,))
            deleted = cur.rowcount > 0
        if deleted:
            try:
                from services.audit import AuditService

                AuditService.log("policy", "policy.deleted", actor="user", summary=f"Deleted policy {policy_id}")
            except Exception:
                pass
        return deleted

    # -------------------------------------------------------------- decision
    @classmethod
    def evaluate(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        risk = context.get("risk_class") or RiskClass.READ_ONLY
        action = context.get("action") or context.get("tool_name") or "unknown"
        autopilot = bool(context.get("autopilot"))
        interactive = bool(context.get("interactive", True))
        reasons: List[str] = []

        from services.safety import SafetyService

        if SafetyService.kill_switch_engaged() and risk != RiskClass.READ_ONLY:
            return cls._decision(Decision.DENY, risk, None, ["Global kill switch is engaged."], context)

        policies = cls.list_policies(enabled_only=True)
        matched: List[Dict[str, Any]] = []
        for policy in policies:
            if risk not in (policy.get("risk_classes") or RiskClass.ALL):
                continue
            if not cls._scope_matches(policy, context):
                continue
            if not cls._conditions_match(policy, context):
                continue
            if not cls._allowlists_match(policy, context):
                continue
            matched.append(policy)

        if not matched:
            if risk == RiskClass.READ_ONLY:
                return cls._decision(Decision.ALLOW, risk, None, ["Read-only action allowed by default."], context)
            if not interactive and autopilot:
                return cls._decision(
                    Decision.DENY, risk, None,
                    ["Autopilot has no pre-authorized policy for this action."], context,
                )
            return cls._decision(
                Decision.REQUIRE_APPROVAL, risk, None,
                ["No allow policy matched; interactive approval required."], context,
            )

        matched.sort(key=lambda p: (-int(p.get("priority", 50)), p["id"]))
        deny = next((p for p in matched if p["effect"] == Decision.DENY), None)
        if deny:
            return cls._decision(Decision.DENY, risk, deny["id"], [f"Denied by policy '{deny['name']}'."], context)

        approval = next((p for p in matched if p["effect"] == Decision.REQUIRE_APPROVAL), None)
        allow = next((p for p in matched if p["effect"] == Decision.ALLOW), None)

        for policy in matched:
            rate = policy.get("rate_limit_per_hour")
            if rate:
                used = cls._rate_used(policy, context)
                if used >= int(rate):
                    return cls._decision(
                        Decision.DENY, risk, policy["id"],
                        [f"Rate limit reached for policy '{policy['name']}' ({used}/{rate} per hour)."], context,
                    )
            spend = policy.get("spend_limit_usd")
            if spend is not None:
                spent = cls._spend_used(context)
                if spent >= float(spend):
                    return cls._decision(
                        Decision.DENY, risk, policy["id"],
                        [f"Spend limit reached for policy '{policy['name']}' (${spent:.2f}/${float(spend):.2f})."], context,
                    )

        if allow:
            if risk in RiskClass.HIGH_RISK and not allow.get("explicit_allow"):
                # A broad/ordinary allow policy cannot silently authorize high-risk actions.
                if approval:
                    return cls._decision(
                        Decision.REQUIRE_APPROVAL, risk, approval["id"],
                        [f"High-risk action requires approval ('{approval['name']}')."], context,
                    )
                return cls._decision(
                    Decision.REQUIRE_APPROVAL, risk, allow["id"],
                    ["High-risk action allowed only with explicit approval or an explicit allow policy."], context,
                )
            if autopilot and not allow.get("explicit_allow"):
                return cls._decision(
                    Decision.DENY, risk, allow["id"],
                    [f"Policy '{allow['name']}' is not pre-authorized for autopilot."], context,
                )
            reasons.append(f"Allowed by policy '{allow['name']}'.")
            return cls._decision(Decision.ALLOW, risk, allow["id"], reasons, context)

        if approval:
            return cls._decision(
                Decision.REQUIRE_APPROVAL, risk, approval["id"],
                [f"Required by policy '{approval['name']}'."], context,
            )
        return cls._decision(
            Decision.REQUIRE_APPROVAL, risk, None, ["No allow policy matched."], context,
        )

    # -------------------------------------------------------------- approvals
    @classmethod
    def request_approval(cls, context: Dict[str, Any], reason: str, payload: Any = None) -> Dict[str, Any]:
        action = context.get("action") or context.get("tool_name") or "unknown"
        tool = context.get("tool_name")
        run_id = context.get("run_id")
        agent_id = context.get("agent_id")

        # Reuse a recent pending approval for the same run/tool to avoid duplicates.
        with db_session() as conn:
            existing = None
            if run_id and tool:
                existing = conn.execute(
                    """
                    SELECT * FROM approvals WHERE status='pending' AND run_id=? AND tool_name=?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (run_id, tool),
                ).fetchone()
            if existing:
                return cls._approval_row(existing)

            approval_id = f"apr_{uuid.uuid4().hex[:12]}"
            now = _now()
            expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
            safe_context = redact_obj(context)
            safe_payload = redact_obj(payload if payload is not None else context)
            if isinstance(payload, dict) and "args" in payload:
                safe_payload = dict(safe_payload)
                safe_payload["args_hash"] = args_hash(payload.get("args"))
            conn.execute(
                """
                INSERT INTO approvals (
                    id, created_at, status, risk_class, action, agent_id, task_id,
                    run_id, tool_name, reason, payload, expires_at
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id, now, context.get("risk_class"), action, agent_id,
                    context.get("task_id"), run_id, tool, redact_text(reason)[:1000],
                    json.dumps(safe_payload, default=str),
                    expires,
                ),
            )
        try:
            from services.audit import AuditService

            AuditService.log(
                "approval", "approval.requested", actor="agent" if agent_id else "system",
                status="approval_required", risk_class=context.get("risk_class"),
                agent_id=agent_id, task_id=context.get("task_id"), run_id=run_id,
                tool_name=tool, approval_id=approval_id, summary=reason,
                data=safe_context, activity_title="Approval requested", severity="warning",
            )
            from services.events import NotificationService

            NotificationService.create(
                title="Approval required",
                body=f"{action}: {reason}",
                severity="warning",
                kind="approval",
                action_url="/platform#approvals",
                data={"approval_id": approval_id, "run_id": run_id},
                activity=False,
            )
        except Exception:
            pass
        return cls.get_approval(approval_id)  # type: ignore[return-value]

    @classmethod
    def get_approval(cls, approval_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        return cls._approval_row(row) if row else None

    @classmethod
    def list_approvals(cls, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        where, params = "", []
        if status:
            where = "WHERE status = ?"
            params.append(status)
        params.append(max(1, min(limit, 500)))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM approvals {where} ORDER BY created_at DESC LIMIT ?", params
            ).fetchall()
        return [cls._approval_row(row) for row in rows]

    @classmethod
    def resolve_approval(
        cls, approval_id: str, approved: bool, note: str = "", actor: str = "user"
    ) -> Optional[Dict[str, Any]]:
        approval = cls.get_approval(approval_id)
        if not approval:
            return None
        if approval["status"] != "pending":
            return approval
        status = "approved" if approved else "denied"
        with db_session() as conn:
            conn.execute(
                "UPDATE approvals SET status=?, resolved_at=?, resolution_note=? WHERE id=?",
                (status, _now(), redact_text(note)[:1000], approval_id),
            )
        try:
            from services.audit import AuditService

            AuditService.log(
                "approval", f"approval.{status}", actor=actor, status=status,
                risk_class=approval.get("risk_class"), agent_id=approval.get("agent_id"),
                task_id=approval.get("task_id"), run_id=approval.get("run_id"),
                tool_name=approval.get("tool_name"), approval_id=approval_id,
                summary=note or f"Approval {status}",
                data={"approval": approval, "note": note},
                activity_title=f"Approval {status}",
                severity="info" if approved else "warning",
            )
        except Exception:
            pass
        return cls.get_approval(approval_id)

    @classmethod
    def approved_for_run(cls, run_id: str, tool_name: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute(
                """
                SELECT * FROM approvals WHERE run_id=? AND tool_name=? AND status='approved'
                ORDER BY resolved_at DESC LIMIT 1
                """,
                (run_id, tool_name),
            ).fetchone()
        return cls._approval_row(row) if row else None

    @classmethod
    def expire_stale(cls, hours: int = 24) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with db_session() as conn:
            cur = conn.execute(
                "UPDATE approvals SET status='expired', resolved_at=? WHERE status='pending' AND created_at < ?",
                (_now(), cutoff),
            )
            return cur.rowcount

    # --------------------------------------------------------------- helpers
    @classmethod
    def _decision(
        cls, decision: str, risk: str, policy_id: Optional[str], reasons: List[str], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "decision": decision,
            "risk_class": risk,
            "policy_id": policy_id,
            "reasons": reasons,
            "action": context.get("action") or context.get("tool_name"),
            "autopilot": bool(context.get("autopilot")),
        }

    @classmethod
    def _scope_matches(cls, policy: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        scope_type = policy.get("scope_type") or "global"
        value = policy.get("scope_value")
        if scope_type == "global":
            return True
        if scope_type == "agent":
            return fnmatch.fnmatch(str(ctx.get("agent_id") or ""), str(value or ""))
        if scope_type == "tool":
            return fnmatch.fnmatch(str(ctx.get("tool_name") or ""), str(value or ""))
        if scope_type == "connector":
            return fnmatch.fnmatch(str(ctx.get("connector_id") or ""), str(value or ""))
        if scope_type == "domain":
            return _domain_matches(str(ctx.get("domain") or ""), [str(value or "")])
        if scope_type == "contact":
            return fnmatch.fnmatch(str(ctx.get("contact") or "").lower(), str(value or "").lower())
        if scope_type == "folder":
            path = str(ctx.get("path") or ctx.get("folder") or "")
            return _path_matches(path, [str(value or "")])
        if scope_type == "action":
            return fnmatch.fnmatch(str(ctx.get("action") or ""), str(value or ""))
        if scope_type == "risk":
            return fnmatch.fnmatch(str(ctx.get("risk_class") or ""), str(value or ""))
        return False

    @classmethod
    def _conditions_match(cls, policy: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        cond = policy.get("conditions") or {}
        if not isinstance(cond, dict):
            return False
        for key, expected in cond.items():
            if key == "autopilot" and bool(ctx.get("autopilot")) != bool(expected):
                return False
            if key == "interactive" and bool(ctx.get("interactive", True)) != bool(expected):
                return False
            if key == "path_prefix":
                if not str(ctx.get("path") or "").startswith(str(expected)):
                    return False
            if key == "tool_in":
                if ctx.get("tool_name") not in (expected or []):
                    return False
            if key == "agent_in":
                if ctx.get("agent_id") not in (expected or []):
                    return False
            if key == "risk_in":
                if ctx.get("risk_class") not in (expected or []):
                    return False
            if key == "domain_in":
                if not _domain_matches(str(ctx.get("domain") or ""), expected or []):
                    return False
            if key == "max_amount_usd" and float(ctx.get("amount_usd") or 0) > float(expected):
                return False
            if key == "time_window":
                try:
                    now = datetime.now(timezone.utc).strftime("%H:%M")
                    if not (str(expected.get("start", "00:00")) <= now <= str(expected.get("end", "23:59"))):
                        return False
                except Exception:
                    return False
        return True

    @classmethod
    def _allowlists_match(cls, policy: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        domains = policy.get("allow_domains") or []
        if domains:
            if not _domain_matches(str(ctx.get("domain") or ""), domains):
                return False
        contacts = policy.get("allow_contacts") or []
        if contacts:
            contact = str(ctx.get("contact") or "").lower()
            if not any(fnmatch.fnmatch(contact, str(c).lower()) for c in contacts):
                return False
        folders = policy.get("allow_folders") or []
        if folders:
            if not _path_matches(str(ctx.get("path") or ctx.get("folder") or ""), folders):
                return False
        return True

    @classmethod
    def _rate_used(cls, policy: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with db_session() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM audit_events
                WHERE created_at > ? AND (
                    (tool_name IS NOT NULL AND tool_name = ?) OR action = ?
                )
                """,
                (cutoff, ctx.get("tool_name"), ctx.get("action")),
            ).fetchone()
        return int(row["n"] or 0)

    @classmethod
    def _spend_used(cls, ctx: Dict[str, Any]) -> float:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with db_session() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM v2_runs WHERE created_at > ? AND (? IS NULL OR agent_id = ?)",
                (cutoff, ctx.get("agent_id"), ctx.get("agent_id")),
            ).fetchone()
        return float(row["total"] or 0.0)

    @classmethod
    def _validate(cls, policy: Dict[str, Any]) -> None:
        if not policy.get("name"):
            raise ValueError("Policy name is required")
        if policy.get("effect") not in EFFECTS:
            raise ValueError(f"effect must be one of {EFFECTS}")
        if policy.get("scope_type", "global") not in SCOPE_TYPES:
            raise ValueError(f"scope_type must be one of {SCOPE_TYPES}")
        risks = policy.get("risk_classes") or []
        for risk in risks:
            if risk not in RiskClass.ALL:
                raise ValueError(f"Unknown risk class: {risk}")
        if policy.get("scope_type") not in (None, "global") and not policy.get("scope_value"):
            raise ValueError("scope_value is required for non-global policies")
        for field in ("conditions",):
            if policy.get(field) is not None and not isinstance(policy.get(field), dict):
                raise ValueError(f"{field} must be an object")
        for field in ("allow_domains", "allow_contacts", "allow_folders"):
            if policy.get(field) is not None and not isinstance(policy.get(field), list):
                raise ValueError(f"{field} must be a list")

    @classmethod
    def _insert_values(cls, policy_id: str, payload: Dict[str, Any], now: str) -> tuple:
        return (
            policy_id, payload["name"], payload.get("description", ""),
            int(bool(payload.get("enabled", True))), int(payload.get("priority", 50)),
            payload["effect"], payload.get("scope_type", "global"), payload.get("scope_value"),
            json.dumps(payload.get("risk_classes") or []), json.dumps(payload.get("conditions") or {}),
            payload.get("rate_limit_per_hour"), payload.get("spend_limit_usd"),
            payload.get("time_limit_seconds"), json.dumps(payload.get("allow_domains") or []),
            json.dumps(payload.get("allow_contacts") or []), json.dumps(payload.get("allow_folders") or []),
            int(bool(payload.get("explicit_allow"))), now, now,
        )

    @classmethod
    def _row(cls, row) -> Dict[str, Any]:
        return {
            "id": row["id"], "name": row["name"], "description": row["description"],
            "enabled": bool(row["enabled"]), "priority": row["priority"],
            "effect": row["effect"], "scope_type": row["scope_type"],
            "scope_value": row["scope_value"],
            "risk_classes": _json(row["risk_classes"], []),
            "conditions": _json(row["conditions"], {}),
            "rate_limit_per_hour": row["rate_limit_per_hour"],
            "spend_limit_usd": row["spend_limit_usd"],
            "time_limit_seconds": row["time_limit_seconds"],
            "allow_domains": _json(row["allow_domains"], []),
            "allow_contacts": _json(row["allow_contacts"], []),
            "allow_folders": _json(row["allow_folders"], []),
            "explicit_allow": bool(row["explicit_allow"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @classmethod
    def _approval_row(cls, row) -> Dict[str, Any]:
        try:
            payload = json.loads(row["payload"] or "{}")
        except Exception:
            payload = {}
        return {
            "id": row["id"], "created_at": row["created_at"],
            "resolved_at": row["resolved_at"], "status": row["status"],
            "risk_class": row["risk_class"], "action": row["action"],
            "agent_id": row["agent_id"], "task_id": row["task_id"], "run_id": row["run_id"],
            "tool_name": row["tool_name"], "reason": row["reason"], "payload": payload,
            "resolution_note": row["resolution_note"], "expires_at": row["expires_at"],
        }
