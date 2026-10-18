"""
Episodic Memory (Tier 2 Memory): Past highlights and semantic retrieval using SQLite FTS5 + Hybrid search.
"""
from typing import List, Dict, Optional
import sqlite3
import re
import struct
from .database import db_session
from persona.chatml import estimate_tokens

_model = None
_model_init_attempted = False

def get_embedding_model():
    """Lazily loads sentence-transformers all-MiniLM-L6-v2 model on CPU."""
    global _model, _model_init_attempted
    if not _model_init_attempted:
        _model_init_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            _model = None
    return _model

def compute_embedding(text: str) -> Optional[bytes]:
    """Computes a unit-normalized float32 embedding vector packed as bytes."""
    model = get_embedding_model()
    if model is None or not text.strip():
        return None
    try:
        vec = model.encode(text.strip(), normalize_embeddings=True)
        return struct.pack(f"{len(vec)}f", *vec)
    except Exception:
        return None

def unpack_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    """Unpacks a raw float32 BLOB into a list of floats."""
    if not blob:
        return None
    try:
        num_floats = len(blob) // 4
        return list(struct.unpack(f"{num_floats}f", blob))
    except Exception:
        return None

def cosine_similarity(u: List[float], v: List[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors."""
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    return max(0.0, min(1.0, (dot + 1.0) / 2.0))


class EpisodicMemory:
    @staticmethod
    def add_memory(session_id: str, content: str, summary: str = "", tags: str = "") -> int:
        """Stores an episodic event/chunk into long-term episodic memory with embedding vector."""
        embed_text = f"{summary}\n{content}".strip() if summary else content.strip()
        embedding_blob = compute_embedding(embed_text)

        with db_session() as conn:
            cursor = conn.execute("""
            INSERT INTO episodic_memories (session_id, content, summary, tags, embedding)
            VALUES (?, ?, ?, ?, ?);
            """, (session_id, content.strip(), summary.strip(), tags.strip(), embedding_blob))
            return cursor.lastrowid

    @classmethod
    def search_memories(cls, query: str, limit: int = 4) -> List[Dict]:
        """Alias for retrieve_relevant for backwards compatibility."""
        return cls.retrieve_relevant(query, limit)

    @classmethod
    def retrieve_relevant(cls, query: str, limit: int = 4) -> List[Dict]:
        """
        Retrieves relevant episodic memories using Hybrid Search:
        0.4 * normalized FTS5 BM25 + 0.6 * dense cosine similarity (all-MiniLM-L6-v2).
        Falls back seamlessly to pure FTS5 BM25 when sentence-transformers is unavailable.
        """
        clean_words = re.findall(r"\w+", query)
        if not clean_words:
            return []

        # Formulate FTS5 match expression: e.g. "medicine OR doctor OR exam"
        fts_query = " OR ".join(clean_words[:8])
        candidates = []

        with db_session() as conn:
            try:
                cursor = conn.execute("""
                SELECT m.id, m.session_id, m.content, m.summary, m.tags, m.embedding, m.created_at,
                       bm25(episodic_fts) as rank
                FROM episodic_fts f
                JOIN episodic_memories m ON f.rowid = m.id
                WHERE episodic_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?;
                """, (fts_query, max(limit * 3, 10)))
                candidates = [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                # Fallback to simple LIKE search if FTS query format fails
                like_pattern = f"%{clean_words[0]}%"
                cursor = conn.execute("""
                SELECT id, session_id, content, summary, tags, embedding, created_at, 0 as rank
                FROM episodic_memories
                WHERE content LIKE ? OR summary LIKE ?
                ORDER BY id DESC
                LIMIT ?;
                """, (like_pattern, like_pattern, max(limit * 3, 10)))
                candidates = [dict(row) for row in cursor.fetchall()]

        if not candidates:
            return []

        # Check if semantic embedding model is available
        query_blob = compute_embedding(query)
        q_vec = unpack_embedding(query_blob) if query_blob else None

        if not q_vec:
            # Pure BM25 mode (sentence-transformers not installed or failed)
            candidates.sort(key=lambda x: x.get("rank", 0))
            return candidates[:limit]

        # Hybrid fusion mode: 0.4 BM25 + 0.6 Cosine Similarity
        # Normalize BM25 ranks: best candidate rank is closest to lowest/most negative
        ranks = [c.get("rank", 0.0) for c in candidates]
        min_r = min(ranks) if ranks else 0.0
        max_r = max(ranks) if ranks else 0.0
        range_r = (max_r - min_r) if (max_r - min_r) > 1e-6 else 1.0

        scored_candidates = []
        for c in candidates:
            # BM25: lower/more negative is better -> invert to [0.0, 1.0] where 1.0 is best
            bm25_score = 1.0 - ((c.get("rank", min_r) - min_r) / range_r)

            # Dense vector cosine similarity
            m_vec = unpack_embedding(c.get("embedding"))
            sem_score = cosine_similarity(q_vec, m_vec) if m_vec else 0.0

            hybrid_score = 0.4 * bm25_score + 0.6 * sem_score
            c["hybrid_score"] = round(hybrid_score, 4)
            scored_candidates.append(c)

        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored_candidates[:limit]

    @classmethod
    def format_for_prompt(cls, query: str, max_tokens: int = 350) -> str:
        """
        Retrieves memories relevant to the query and formats into a compact prompt block.
        """
        memories = cls.retrieve_relevant(query, limit=3)
        if not memories:
            return ""

        lines = [
            "[PAST CONVERSATIONS YOU RECALL WITH THEM]",
            "Things you talked about earlier (bring up naturally if relevant):"
        ]
        current_tokens = estimate_tokens(lines[0] + lines[1])

        for m in memories:
            text = m["summary"] if m["summary"] else m["content"]
            # Truncate text if excessively long
            snippet = (text[:180] + "...") if len(text) > 180 else text
            line = f"- {snippet}"
            line_tokens = estimate_tokens(line)

            if current_tokens + line_tokens > max_tokens:
                break
            lines.append(line)
            current_tokens += line_tokens

        if len(lines) <= 2:
            return ""
        return "\n".join(lines)
