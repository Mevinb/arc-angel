"""
Version 2 platform schema.

The original chatbot tables live in ``memory.database``.  This module adds the
modular service-layer tables used by the autonomous platform: agents, goals,
runs, tools/audit, policies, approvals, knowledge, connectors, automations,
notifications, mobile devices, plugins and local finance.

The schema is intentionally additive and idempotent so existing installations
migrate in place without losing chats, memory, or task history.
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Iterable

from memory.database import db_session

_init_lock = threading.Lock()
_initialized = False


def _add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    except sqlite3.OperationalError:
        pass


def init_platform_db(force: bool = False) -> None:
    """Create v2 tables/indexes and migrate existing tables. Safe to call often."""
    global _initialized
    if _initialized and not force:
        return
    with _init_lock:
        if _initialized and not force:
            return
        with db_session() as conn:
            cur = conn.cursor()

            # --------------------------------------------------------- chats v2
            for col, ddl in (
                ("branch_of", "TEXT"),
                ("branch_from_message_id", "INTEGER"),
                ("private", "INTEGER DEFAULT 0"),
                ("profile", "TEXT DEFAULT 'personal'"),
                ("pinned", "INTEGER DEFAULT 0"),
                ("archived", "INTEGER DEFAULT 0"),
                ("model_profile", "TEXT DEFAULT 'balanced'"),
                ("metadata", "TEXT"),
            ):
                _add_column(cur, "sessions", col, ddl)

            _add_column(cur, "user_dossier", "profile", "TEXT DEFAULT 'personal'")
            _add_column(cur, "user_dossier", "source", "TEXT DEFAULT 'chat'")
            _add_column(cur, "user_dossier", "last_used_at", "TIMESTAMP")
            _add_column(cur, "episodic_memories", "profile", "TEXT DEFAULT 'personal'")
            _add_column(cur, "episodic_memories", "private", "INTEGER DEFAULT 0")
            _add_column(cur, "episodic_memories", "confidence", "REAL DEFAULT 1.0")

            # ---------------------------------------------------- audit trail
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk_class TEXT,
                    agent_id TEXT,
                    task_id TEXT,
                    run_id TEXT,
                    tool_name TEXT,
                    connector_id TEXT,
                    approval_id TEXT,
                    summary TEXT,
                    data TEXT,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL UNIQUE
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_events(run_id, id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_events(category, id);")
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
                """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
                """
            )

            # ------------------------------------------- unified activity feed
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    detail TEXT,
                    ref_type TEXT,
                    ref_id TEXT,
                    agent_id TEXT,
                    severity TEXT DEFAULT 'info',
                    data TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_events(id DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_activity_kind ON activity_events(kind, id DESC);")

            # ------------------------------------------------- notifications
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT,
                    severity TEXT DEFAULT 'info',
                    kind TEXT DEFAULT 'system',
                    read INTEGER DEFAULT 0,
                    dismissed INTEGER DEFAULT 0,
                    action_url TEXT,
                    data TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications(read, dismissed, id DESC);")

            # ------------------------------------------- encrypted credential vault
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS secret_store (
                    name TEXT PRIMARY KEY,
                    ciphertext TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # ------------------------------------------------------- policies
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS policies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    enabled INTEGER DEFAULT 1,
                    priority INTEGER DEFAULT 50,
                    effect TEXT NOT NULL,
                    scope_type TEXT DEFAULT 'global',
                    scope_value TEXT,
                    risk_classes TEXT,
                    conditions TEXT,
                    rate_limit_per_hour INTEGER,
                    spend_limit_usd REAL,
                    time_limit_seconds INTEGER,
                    allow_domains TEXT,
                    allow_contacts TEXT,
                    allow_folders TEXT,
                    explicit_allow INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_policies_enabled ON policies(enabled, priority DESC);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    status TEXT NOT NULL,
                    risk_class TEXT,
                    action TEXT,
                    agent_id TEXT,
                    task_id TEXT,
                    run_id TEXT,
                    tool_name TEXT,
                    reason TEXT,
                    payload TEXT,
                    resolution_note TEXT,
                    expires_at TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, created_at DESC);")

            # --------------------------------------------------------- agents
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    role TEXT,
                    system_prompt TEXT,
                    tools TEXT,
                    risk_profile TEXT DEFAULT 'balanced',
                    model_profile TEXT DEFAULT 'balanced',
                    icon TEXT DEFAULT '🤖',
                    is_builtin INTEGER DEFAULT 0,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # ---------------------------------------------------------- tasks
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    agent_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    priority INTEGER DEFAULT 3,
                    due_at TEXT,
                    parent_id TEXT,
                    dependencies TEXT,
                    budget TEXT,
                    policy_id TEXT,
                    workspace_dir TEXT,
                    session_id TEXT,
                    origin TEXT DEFAULT 'user',
                    tags TEXT,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_v2_tasks_status ON v2_tasks(status, priority DESC, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_v2_tasks_agent ON v2_tasks(agent_id);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS v2_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    agent_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempt INTEGER DEFAULT 1,
                    checkpoint TEXT,
                    policy_id TEXT,
                    budget TEXT,
                    model TEXT,
                    provider TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    heartbeat_at TEXT,
                    error TEXT,
                    result TEXT,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    replay_of TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_v2_runs_task ON v2_runs(task_id, created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_v2_runs_status ON v2_runs(status);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS run_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    tool_name TEXT,
                    input TEXT,
                    output TEXT,
                    error TEXT,
                    checkpoint TEXT,
                    started_at TEXT,
                    finished_at TEXT
                );
                """
            )
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_run_steps_idx ON run_steps(run_id, idx);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_run_events ON run_events(run_id, id);")

            # ------------------------------------------------------- knowledge
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_sources (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    uri TEXT,
                    author TEXT,
                    tags TEXT,
                    metadata TEXT,
                    content_hash TEXT,
                    char_count INTEGER DEFAULT 0,
                    private INTEGER DEFAULT 0,
                    profile TEXT DEFAULT 'personal',
                    credibility REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sources_type ON knowledge_sources(source_type, created_at DESC);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_estimate INTEGER DEFAULT 0,
                    embedding TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source ON knowledge_chunks(source_id, idx);")

            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                        chunk_id UNINDEXED,
                        source_id UNINDEXED,
                        title,
                        content,
                        tags,
                        tokenize='porter unicode61'
                    );
                    """
                )
            except sqlite3.OperationalError:
                # FTS5 is present on standard sqlite builds; degrade to LIKE search.
                pass

            # -------------------------------------------------------- research
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT DEFAULT 'draft',
                    task_id TEXT,
                    report TEXT,
                    bibliography TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'note',
                    title TEXT,
                    content TEXT,
                    url TEXT,
                    credibility REAL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    status TEXT DEFAULT 'unverified',
                    support TEXT,
                    source_url TEXT,
                    conflict TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

            # ------------------------------------------------------ connectors
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS connector_connections (
                    id TEXT PRIMARY KEY,
                    connector_id TEXT NOT NULL,
                    account_label TEXT,
                    status TEXT NOT NULL DEFAULT 'connected',
                    scopes TEXT,
                    metadata TEXT,
                    access_token_secret TEXT,
                    refresh_token_secret TEXT,
                    expires_at TEXT,
                    last_sync_at TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS connector_states (
                    state TEXT PRIMARY KEY,
                    connector_id TEXT NOT NULL,
                    redirect_uri TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS connector_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    connection_id TEXT NOT NULL,
                    connector_id TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    item_type TEXT NOT NULL DEFAULT 'item',
                    title TEXT,
                    body TEXT,
                    author TEXT,
                    url TEXT,
                    occurred_at TEXT,
                    content_hash TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(connection_id, external_id)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_connector_items ON connector_items(connector_id, occurred_at DESC);")
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_connector_hash ON connector_items(connection_id, content_hash);"
            )

            # ----------------------------------------------------- automations
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS automations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    enabled INTEGER DEFAULT 1,
                    trigger_type TEXT NOT NULL,
                    trigger_config TEXT,
                    action_type TEXT DEFAULT 'agent_task',
                    action_config TEXT,
                    agent_id TEXT,
                    policy_id TEXT,
                    next_run_at TEXT,
                    last_run_at TEXT,
                    last_status TEXT,
                    run_count INTEGER DEFAULT 0,
                    webhook_secret TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS automation_states (
                    automation_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (automation_id, key)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS automation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    automation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    task_id TEXT
                );
                """
            )

            # ------------------------------------------- prompt library/shortcuts
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_library (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    tags TEXT,
                    kind TEXT DEFAULT 'prompt',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_shortcuts (
                    id TEXT PRIMARY KEY,
                    command TEXT NOT NULL UNIQUE,
                    expansion TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT DEFAULT 'personal',
                    settings TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # ---------------------------------------------- mobile companion
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS mobile_devices (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    platform TEXT,
                    pairing_code_hash TEXT,
                    token_hash TEXT,
                    paired_at TEXT,
                    last_seen TEXT,
                    push_enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                """
            )

            # --------------------------------------------------------- plugins
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS plugins (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT,
                    description TEXT,
                    path TEXT,
                    manifest TEXT,
                    enabled INTEGER DEFAULT 1,
                    installed_at TEXT NOT NULL
                );
                """
            )

            # ------------------------------------------- code/files checkpoints
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS file_checkpoints (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    root TEXT NOT NULL,
                    storage_dir TEXT NOT NULL,
                    files_count INTEGER DEFAULT 0,
                    size_bytes INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS code_repos (
                    id TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    symbol_count INTEGER DEFAULT 0,
                    index_data TEXT,
                    indexed_at TEXT NOT NULL
                );
                """
            )

            # --------------------------------------------- model performance/votes
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_stats (
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    task_type TEXT DEFAULT 'general',
                    runs INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    total_latency_ms REAL DEFAULT 0,
                    tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    votes_up INTEGER DEFAULT 0,
                    votes_down INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model, provider, task_type)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    vote TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

            # -------------------------------------------------------- documents
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'note',
                    content TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # --------------------------------------------------------- finance
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS finance_transactions (
                    id TEXT PRIMARY KEY,
                    occurred_on TEXT NOT NULL,
                    description TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    category TEXT,
                    account TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    amount REAL,
                    due_on TEXT,
                    cadence TEXT DEFAULT 'monthly',
                    paid INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

            # ----------------------------------------------------------- habits
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS habits (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    cadence TEXT DEFAULT 'daily',
                    target INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS habit_logs (
                    habit_id TEXT NOT NULL,
                    day TEXT NOT NULL,
                    value INTEGER DEFAULT 1,
                    PRIMARY KEY (habit_id, day)
                );
                """
            )

            # ---------------------------------------------------- platform meta
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS platform_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )

        _initialized = True


def reset_init_state() -> None:
    """Used by tests after pointing GROK_DB_PATH at a fresh database."""
    global _initialized
    _initialized = False


PLATFORM_TABLES: Iterable[str] = (
    "v2_agents", "v2_tasks", "v2_runs", "run_steps", "run_events",
    "knowledge_sources", "knowledge_chunks", "research_projects",
    "research_notes", "research_claims", "connector_connections",
    "connector_items", "automations", "automation_runs", "notifications",
    "activity_events", "prompt_library", "documents", "finance_transactions",
    "bills", "habits", "habit_logs", "mobile_devices", "plugins",
)
