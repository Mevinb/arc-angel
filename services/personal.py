"""Dashboard, documents, finance, bills, habits, prompts, shortcuts and profiles."""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.audit import AuditService
from services.events import ActivityService, NotificationService
from services.knowledge import KnowledgeService
from services.tasks import TaskService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersonalService:
    # --------------------------------------------------------------- dashboard
    @classmethod
    def dashboard(cls) -> Dict[str, Any]:
        today = date.today().isoformat()
        with db_session() as conn:
            task_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM v2_tasks GROUP BY status"
            ).fetchall()
            run_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM v2_runs GROUP BY status"
            ).fetchall()
            usage = conn.execute(
                "SELECT COALESCE(SUM(tokens_in+tokens_out),0) AS tokens, COALESCE(SUM(cost_usd),0) AS cost FROM v2_runs"
            ).fetchone()
            knowledge = conn.execute(
                "SELECT COUNT(*) AS sources, COALESCE(SUM(char_count),0) AS chars FROM knowledge_sources"
            ).fetchone()
            unread = conn.execute(
                "SELECT COUNT(*) AS n FROM notifications WHERE read=0 AND dismissed=0"
            ).fetchone()["n"]
            approvals = conn.execute(
                "SELECT COUNT(*) AS n FROM approvals WHERE status='pending'"
            ).fetchone()["n"]
            due_bills = conn.execute(
                "SELECT * FROM bills WHERE paid=0 AND due_on<=? ORDER BY due_on LIMIT 10",
                ((date.today() + timedelta(days=7)).isoformat(),),
            ).fetchall()
            habits = conn.execute("SELECT * FROM habits ORDER BY created_at LIMIT 20").fetchall()
            habit_logs = conn.execute(
                "SELECT habit_id, value FROM habit_logs WHERE day=?", (today,)
            ).fetchall()
            activity = conn.execute(
                "SELECT * FROM activity_events ORDER BY id DESC LIMIT 12"
            ).fetchall()
            upcoming = conn.execute(
                "SELECT id, title, status, due_at, priority FROM v2_tasks WHERE status IN ('queued','waiting','running','awaiting_approval','paused','interrupted') ORDER BY priority DESC, created_at DESC LIMIT 10"
            ).fetchall()
        logs = {r["habit_id"]: r["value"] for r in habit_logs}
        habit_out = []
        for row in habits:
            value = logs.get(row["id"], 0)
            target = row["target"] or 1
            habit_out.append({
                "id": row["id"], "name": row["name"], "cadence": row["cadence"],
                "target": target, "today": value, "done": value >= target,
            })
        return {
            "date": today,
            "tasks": {r["status"]: r["n"] for r in task_rows},
            "runs": {r["status"]: r["n"] for r in run_rows},
            "tokens": usage["tokens"], "cost_usd": round(usage["cost"] or 0, 6),
            "knowledge": {"sources": knowledge["sources"], "chars": knowledge["chars"]},
            "notifications_unread": unread, "pending_approvals": approvals,
            "bills_due_soon": [dict(r) for r in due_bills],
            "habits": habit_out,
            "upcoming_tasks": [dict(r) for r in upcoming],
            "recent_activity": [
                {
                    "id": r["id"], "created_at": r["created_at"], "kind": r["kind"],
                    "title": r["title"], "detail": r["detail"], "severity": r["severity"],
                }
                for r in activity
            ],
        }

    @classmethod
    def daily_brief(cls) -> Dict[str, Any]:
        data = cls.dashboard()
        lines = [f"# Daily brief — {data['date']}", ""]
        if data["upcoming_tasks"]:
            lines.append("## Priorities")
            for task in data["upcoming_tasks"][:5]:
                lines.append(f"- [{task['status']}] {task['title']} (priority {task['priority']})")
        else:
            lines.append("No active tasks.")
        if data["pending_approvals"]:
            lines.append(f"\n**{data['pending_approvals']} approval(s) waiting.**")
        if data["bills_due_soon"]:
            lines.append("\n## Bills due soon")
            for bill in data["bills_due_soon"]:
                lines.append(f"- {bill['name']}: {bill['amount']} due {bill['due_on']}")
        pending_habits = [h["name"] for h in data["habits"] if not h["done"]]
        if pending_habits:
            lines.append(f"\n## Habits\nStill open: {', '.join(pending_habits)}")
        return {"markdown": "\n".join(lines), "data": data}

    # ----------------------------------------------------------------- documents
    @classmethod
    def list_documents(cls, q: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        clause, params = "", []
        if q:
            clause = "WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?"
            params = [f"%{q}%"] * 3
        params.append(max(1, min(limit, 300)))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents {clause} ORDER BY updated_at DESC LIMIT ?", params
            ).fetchall()
        return [cls._document_row(r) for r in rows]

    @classmethod
    def get_document(cls, doc_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return cls._document_row(row) if row else None

    @classmethod
    def create_document(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        title = (payload.get("title") or "Untitled").strip()[:300]
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        now = _now()
        with db_session() as conn:
            conn.execute(
                "INSERT INTO documents (id, title, kind, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, title, (payload.get("kind") or "note")[:40], payload.get("content") or "",
                 json.dumps(payload.get("tags") or []), now, now),
            )
        AuditService.log("document", "document.created", actor="user", summary=f"Document created: {title}")
        return cls.get_document(doc_id)  # type: ignore[return-value]

    @classmethod
    def update_document(cls, doc_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not cls.get_document(doc_id):
            return None
        allowed = {"title", "kind", "content", "tags"}
        sets, params = [], []
        for key, value in patch.items():
            if key not in allowed:
                continue
            if key == "tags":
                value = json.dumps(value)
            sets.append(f"{key}=?")
            params.append(value)
        if not sets:
            return cls.get_document(doc_id)
        sets.append("updated_at=?")
        params.extend([_now(), doc_id])
        with db_session() as conn:
            conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE id=?", params)
        return cls.get_document(doc_id)

    @classmethod
    def delete_document(cls, doc_id: str) -> bool:
        with db_session() as conn:
            count = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
        if count:
            AuditService.log("document", "document.deleted", actor="user", summary=f"Document deleted: {doc_id}")
        return bool(count)

    # ------------------------------------------------------------------- finance
    @classmethod
    def list_transactions(
        cls, *, month: Optional[str] = None, category: Optional[str] = None, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if month:
            clauses.append("occurred_on LIKE ?")
            params.append(f"{month}%")
        if category:
            clauses.append("category=?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 1000)))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM finance_transactions {where} ORDER BY occurred_on DESC, id DESC LIMIT ?", params
            ).fetchall()
        return [cls._transaction_row(r) for r in rows]

    @classmethod
    def add_transaction(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            amount = float(payload.get("amount"))
        except (TypeError, ValueError):
            raise ValueError("amount must be a number")
        occurred_on = payload.get("occurred_on") or date.today().isoformat()
        with db_session() as conn:
            cur = conn.execute(
                """
                INSERT INTO finance_transactions
                (occurred_on, description, amount, currency, category, account, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    occurred_on, (payload.get("description") or "")[:500], amount,
                    (payload.get("currency") or "USD").upper()[:8],
                    (payload.get("category") or "uncategorized")[:80],
                    (payload.get("account") or "")[:80], json.dumps(payload.get("metadata") or {}), _now(),
                ),
            )
            row = conn.execute("SELECT * FROM finance_transactions WHERE id=?", (cur.lastrowid,)).fetchone()
        AuditService.log("finance", "transaction.added", actor="user", summary=f"Transaction added: {row['description'][:120]}")
        return cls._transaction_row(row)

    @classmethod
    def delete_transaction(cls, tx_id: int) -> bool:
        with db_session() as conn:
            count = conn.execute("DELETE FROM finance_transactions WHERE id=?", (tx_id,)).rowcount
        return bool(count)

    @classmethod
    def finance_summary(cls, month: Optional[str] = None) -> Dict[str, Any]:
        month = month or date.today().strftime("%Y-%m")
        with db_session() as conn:
            rows = conn.execute(
                "SELECT category, COALESCE(SUM(amount),0) AS total, COUNT(*) AS n FROM finance_transactions WHERE occurred_on LIKE ? GROUP BY category ORDER BY total ASC",
                (f"{month}%",),
            ).fetchall()
            total = conn.execute(
                "SELECT COALESCE(SUM(amount),0) AS total FROM finance_transactions WHERE occurred_on LIKE ?",
                (f"{month}%",),
            ).fetchone()["total"]
        return {
            "month": month, "total": round(total, 2),
            "categories": [{"category": r["category"], "total": round(r["total"], 2), "count": r["n"]} for r in rows],
        }

    @classmethod
    def list_bills(cls, unpaid_only: bool = False) -> List[Dict[str, Any]]:
        where = "WHERE paid=0" if unpaid_only else ""
        with db_session() as conn:
            rows = conn.execute(f"SELECT * FROM bills {where} ORDER BY due_on ASC").fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def create_bill(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            amount = float(payload.get("amount"))
        except (TypeError, ValueError):
            raise ValueError("amount must be a number")
        bill_id = f"bill_{uuid.uuid4().hex[:10]}"
        with db_session() as conn:
            conn.execute(
                "INSERT INTO bills (id, name, amount, due_on, cadence, paid, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                (bill_id, (payload.get("name") or "Bill")[:200], amount,
                 payload.get("due_on") or date.today().isoformat(),
                 (payload.get("cadence") or "monthly")[:30], _now()),
            )
        return {"id": bill_id}

    @classmethod
    def pay_bill(cls, bill_id: str, paid: bool = True) -> bool:
        with db_session() as conn:
            count = conn.execute("UPDATE bills SET paid=? WHERE id=?", (1 if paid else 0, bill_id)).rowcount
        return bool(count)

    # -------------------------------------------------------------------- habits
    @classmethod
    def list_habits(cls) -> List[Dict[str, Any]]:
        today = date.today().isoformat()
        with db_session() as conn:
            habits = conn.execute("SELECT * FROM habits ORDER BY created_at").fetchall()
            logs = conn.execute(
                "SELECT habit_id, day, value FROM habit_logs WHERE day >= ?", ((date.today() - timedelta(days=60)).isoformat(),)
            ).fetchall()
        by_habit: Dict[str, Dict[str, float]] = {}
        for log in logs:
            by_habit.setdefault(log["habit_id"], {})[log["day"]] = log["value"]
        out = []
        for habit in habits:
            days = by_habit.get(habit["id"], {})
            streak = 0
            cursor = date.today()
            while str(cursor) in days and days[str(cursor)] >= habit["target"]:
                streak += 1
                cursor -= timedelta(days=1)
            out.append({
                "id": habit["id"], "name": habit["name"], "cadence": habit["cadence"],
                "target": habit["target"], "today": days.get(today, 0),
                "streak": streak, "last_30": [{"day": str(date.today() - timedelta(days=i)), "value": days.get(str(date.today() - timedelta(days=i)), 0)} for i in range(30)],
            })
        return out

    @classmethod
    def create_habit(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        habit_id = f"h_{uuid.uuid4().hex[:10]}"
        with db_session() as conn:
            conn.execute(
                "INSERT INTO habits (id, name, cadence, target, created_at) VALUES (?, ?, ?, ?, ?)",
                (habit_id, (payload.get("name") or "Habit")[:200],
                 (payload.get("cadence") or "daily")[:30],
                 max(1, int(payload.get("target") or 1)), _now()),
            )
        return {"id": habit_id}

    @classmethod
    def log_habit(cls, habit_id: str, value: float = 1, day: Optional[str] = None) -> Dict[str, Any]:
        day = day or date.today().isoformat()
        with db_session() as conn:
            exists = conn.execute("SELECT 1 FROM habits WHERE id=?", (habit_id,)).fetchone()
            if not exists:
                raise ValueError("Habit not found")
            conn.execute(
                "INSERT INTO habit_logs (habit_id, day, value) VALUES (?, ?, ?) ON CONFLICT(habit_id, day) DO UPDATE SET value=excluded.value",
                (habit_id, day, float(value)),
            )
        return {"ok": True, "habit_id": habit_id, "day": day, "value": float(value)}

    # ------------------------------------------------------------- prompts/shortcuts
    @classmethod
    def list_prompts(cls, category: Optional[str] = None) -> List[Dict[str, Any]]:
        where, params = "", []
        if category:
            where = "WHERE category=?"
            params.append(category)
        with db_session() as conn:
            rows = conn.execute(f"SELECT * FROM prompt_library {where} ORDER BY updated_at DESC", params).fetchall()
        out = []
        for r in rows:
            try:
                tags = json.loads(r["tags"] or "[]")
            except Exception:
                tags = []
            out.append({**dict(r), "tags": tags})
        return out

    @classmethod
    def create_prompt(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt_id = f"pr_{uuid.uuid4().hex[:10]}"
        now = _now()
        with db_session() as conn:
            conn.execute(
                "INSERT INTO prompt_library (id, title, content, category, tags, kind, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (prompt_id, (payload.get("title") or "Prompt")[:200], payload.get("content") or "",
                 (payload.get("category") or "general")[:60], json.dumps(payload.get("tags") or []),
                 (payload.get("kind") or "prompt")[:30], now, now),
            )
        return {"id": prompt_id}

    @classmethod
    def delete_prompt(cls, prompt_id: str) -> bool:
        with db_session() as conn:
            count = conn.execute("DELETE FROM prompt_library WHERE id=?", (prompt_id,)).rowcount
        return bool(count)

    @classmethod
    def list_shortcuts(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute("SELECT * FROM personal_shortcuts ORDER BY command").fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def upsert_shortcut(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        command = (payload.get("command") or "").strip()
        if not command:
            raise ValueError("command is required")
        shortcut_id = payload.get("id") or f"sc_{uuid.uuid4().hex[:8]}"
        with db_session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO personal_shortcuts (id, command, expansion, description, created_at) VALUES (?, ?, ?, ?, ?)",
                (shortcut_id, command[:100], payload.get("expansion") or "", (payload.get("description") or "")[:300], _now()),
            )
        return {"id": shortcut_id}

    @classmethod
    def delete_shortcut(cls, shortcut_id: str) -> bool:
        with db_session() as conn:
            count = conn.execute("DELETE FROM personal_shortcuts WHERE id=?", (shortcut_id,)).rowcount
        return bool(count)

    # ------------------------------------------------------------------- profiles
    @classmethod
    def list_profiles(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute("SELECT * FROM profiles ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            try:
                settings = json.loads(r["settings"] or "{}")
            except Exception:
                settings = {}
            out.append({"id": r["id"], "name": r["name"], "kind": r["kind"], "settings": settings, "created_at": r["created_at"], "updated_at": r["updated_at"]})
        return out

    @classmethod
    def create_profile(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = payload.get("id") or f"profile_{uuid.uuid4().hex[:8]}"
        with db_session() as conn:
            conn.execute(
                "INSERT INTO profiles (id, name, kind, settings, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (profile_id, (payload.get("name") or "Profile")[:100],
                 (payload.get("kind") or "personal")[:40], json.dumps(payload.get("settings") or {}), _now(), _now()),
            )
        return {"id": profile_id}

    @classmethod
    def update_profile(cls, profile_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sets, params = [], []
        if "name" in patch:
            sets.append("name=?")
            params.append(str(patch["name"])[:100])
        if "settings" in patch:
            sets.append("settings=?")
            params.append(json.dumps(patch["settings"] or {}))
        if not sets:
            return next((p for p in cls.list_profiles() if p["id"] == profile_id), None)
        sets.append("updated_at=?")
        params.extend([_now(), profile_id])
        with db_session() as conn:
            conn.execute(f"UPDATE profiles SET {', '.join(sets)} WHERE id=?", params)
        return next((p for p in cls.list_profiles() if p["id"] == profile_id), None)

    @classmethod
    def _document_row(cls, row) -> Dict[str, Any]:
        try:
            tags = json.loads(row["tags"] or "[]")
        except Exception:
            tags = []
        return {
            "id": row["id"], "title": row["title"], "kind": row["kind"],
            "content": row["content"], "tags": tags,
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @classmethod
    def _transaction_row(cls, row) -> Dict[str, Any]:
        try:
            meta = json.loads(row["metadata"] or "{}")
        except Exception:
            meta = {}
        return {
            "id": row["id"], "occurred_on": row["occurred_on"], "description": row["description"],
            "amount": row["amount"], "currency": row["currency"], "category": row["category"],
            "account": row["account"], "metadata": meta, "created_at": row["created_at"],
        }
