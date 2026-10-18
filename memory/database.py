"""
SQLite Database manager with WAL mode for persistent memory and chat storage.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from config import DB_PATH

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

@contextmanager
def db_session():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initializes tables and indexes if not already present."""
    with db_session() as conn:
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            mode TEXT DEFAULT 'classic',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Messages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tokens INTEGER DEFAULT 0,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);")
        
        # Add image_url column if older db schema exists
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN image_url TEXT;")
        except sqlite3.OperationalError:
            pass
        
        # User Dossier (Tier 1: Permanent Entity & Persona Memory)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_dossier (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            confidence REAL DEFAULT 1.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Episodic Memories (Tier 2: Past highlights and conversation chunks)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            tags TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Add embedding column if older schema exists
        try:
            cursor.execute("ALTER TABLE episodic_memories ADD COLUMN embedding BLOB;")
        except sqlite3.OperationalError:
            pass
        
        # FTS5 Full-Text Search virtual table for episodic memory
        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
            content,
            summary,
            tags,
            content='episodic_memories',
            content_rowid='id'
        );
        """)
        
        # Triggers to keep FTS index synchronized
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS episodic_ai AFTER INSERT ON episodic_memories BEGIN
            INSERT INTO episodic_fts(rowid, content, summary, tags) 
            VALUES (new.id, new.content, new.summary, new.tags);
        END;
        """)
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS episodic_ad AFTER DELETE ON episodic_memories BEGIN
            INSERT INTO episodic_fts(episodic_fts, rowid, content, summary, tags) 
            VALUES('delete', old.id, old.content, old.summary, old.tags);
        END;
        """)
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS episodic_au AFTER UPDATE ON episodic_memories BEGIN
            INSERT INTO episodic_fts(episodic_fts, rowid, content, summary, tags) 
            VALUES('delete', old.id, old.content, old.summary, old.tags);
            INSERT INTO episodic_fts(rowid, content, summary, tags) 
            VALUES (new.id, new.content, new.summary, new.tags);
        END;
        """)
        
        # Session Rolling Summaries (Tier 3: Compressed long conversations)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            last_message_id INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """)

# Initialize database on module load
init_db()
