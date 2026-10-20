"""
Personal knowledge vault with hybrid (keyword + semantic + metadata) retrieval.

Sources include local files, folders, notes, URLs, transcripts, emails and
bookmarks.  Every result returns the exact source passage so answers can be
grounded and audited.  Embeddings are deterministic local hashed vectors, so the
vault works fully offline without a model download; sentence-transformers can be
plugged in later without a schema change.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from config import MAX_ATTACHMENT_CHARS
from memory.database import db_session
from services.crypto import redact_text

EMBED_DIM = 384
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".ini", ".cfg", ".html", ".htm",
    ".css", ".sh", ".bash", ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs",
    ".sql", ".log", ".rst", ".tex", ".xml",
}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cache", "dist", "build"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]{2,}", (text or "").lower())[:4000]


def embed_text(text: str) -> List[float]:
    vec = [0.0] * EMBED_DIM
    tokens = _tokens(text)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % EMBED_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
        # small char-trigram signal improves fuzzy matches
        for i in range(max(0, len(token) - 3)):
            tri = token[i:i + 3]
            d = hashlib.blake2b(tri.encode("utf-8"), digest_size=4).digest()
            vec[int.from_bytes(d, "little") % EMBED_DIM] += 0.25 * sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 5) for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _fts_query(query: str) -> str:
    tokens = _tokens(query)[:14]
    return " OR ".join(f'"{t}"' for t in tokens)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip()
        else:
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


class KnowledgeService:
    # ------------------------------------------------------------------ ingest
    @classmethod
    def add_text(
        cls,
        *,
        title: str,
        content: str,
        source_type: str = "note",
        uri: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        private: bool = False,
        profile: str = "personal",
        credibility: float = 0.6,
    ) -> Dict[str, Any]:
        content = (content or "").strip()
        if not content:
            raise ValueError("Cannot add an empty knowledge source")
        content = content[:2_000_000]
        content_hash = _hash(content)
        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()][:30]

        with db_session() as conn:
            existing = conn.execute(
                "SELECT * FROM knowledge_sources WHERE content_hash = ? AND source_type = ? LIMIT 1",
                (content_hash, source_type),
            ).fetchone()
            if existing:
                return {**cls._source_row(existing), "chunk_count": cls._chunk_count(existing["id"]), "deduplicated": True}

            source_id = f"kn_{uuid.uuid4().hex[:12]}"
            now = _now()
            conn.execute(
                """
                INSERT INTO knowledge_sources (
                    id, title, source_type, uri, author, tags, metadata, content_hash,
                    char_count, private, profile, credibility, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id, redact_text(title or "Untitled")[:300], source_type,
                    uri, author, json.dumps(tag_list), json.dumps(metadata or {}),
                    content_hash, len(content), int(bool(private)), profile,
                    float(credibility), now, now,
                ),
            )
            chunks = chunk_text(content)
            for idx, chunk in enumerate(chunks):
                cur = conn.execute(
                    """
                    INSERT INTO knowledge_chunks (source_id, idx, content, token_estimate, embedding, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id, idx, chunk, max(1, len(chunk) // 4),
                        json.dumps(embed_text(chunk)), json.dumps({"title": title, "source_type": source_type}),
                        now,
                    ),
                )
                try:
                    conn.execute(
                        "INSERT INTO knowledge_fts (chunk_id, source_id, title, content, tags) VALUES (?, ?, ?, ?, ?)",
                        (cur.lastrowid, source_id, title or "", chunk, " ".join(tag_list)),
                    )
                except Exception:
                    pass
        try:
            from services.audit import AuditService

            AuditService.log(
                "knowledge", "knowledge.ingested", actor="user",
                summary=f"Ingested '{title}' ({len(chunks)} chunks)",
                data={"source_id": source_id, "source_type": source_type, "uri": uri},
            )
        except Exception:
            pass
        return {
            "id": source_id, "title": title, "source_type": source_type,
            "chunk_count": len(chunks), "content_hash": content_hash,
            "private": private, "profile": profile, "deduplicated": False,
        }

    @classmethod
    def add_note(cls, title: str, content: str, tags: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        return cls.add_text(title=title, content=content, source_type="note", tags=tags, **kwargs)

    @classmethod
    def add_file(
        cls, path: str, title: Optional[str] = None, tags: Optional[List[str]] = None,
        private: bool = False, profile: str = "personal",
    ) -> Dict[str, Any]:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        if p.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("File is larger than 25 MB")
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            try:
                import pypdf

                reader = pypdf.PdfReader(str(p))
                content = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception as exc:
                raise ValueError(f"PDF extraction unavailable: {exc}") from exc
        elif suffix in TEXT_EXTENSIONS:
            content = p.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported file type for knowledge ingestion: {suffix or 'unknown'}")
        return cls.add_text(
            title=title or p.name, content=content[:MAX_ATTACHMENT_CHARS * 20],
            source_type="file", uri=str(p), tags=tags,
            metadata={"path": str(p), "suffix": suffix, "size": p.stat().st_size},
            private=private, profile=profile,
        )

    @classmethod
    def add_folder(
        cls, path: str, tags: Optional[List[str]] = None, max_files: int = 500,
        private: bool = False, profile: str = "personal",
    ) -> Dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        added, skipped = [], []
        count = 0
        for item in root.rglob("*"):
            if count >= max_files:
                break
            if any(part in SKIP_DIRS for part in item.parts):
                continue
            if not item.is_file() or item.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                added.append(cls.add_file(str(item), tags=tags, private=private, profile=profile))
                count += 1
            except Exception as exc:
                skipped.append({"path": str(item), "error": str(exc)})
        return {"root": str(root), "ingested": len(added), "skipped": skipped[:50], "sources": added[:200]}

    @classmethod
    async def add_url(
        cls, url: str, tags: Optional[List[str]] = None, private: bool = False,
        profile: str = "personal",
    ) -> Dict[str, Any]:
        from engine.web_tools import web_fetch
        from services.builtin_tools import is_safe_public_url

        if not is_safe_public_url(url, allow_private=__import__("os").getenv("GROK_ALLOW_PRIVATE_FETCH", "0") == "1"):
            raise ValueError("URL blocked: only public http(s) hosts are allowed")
        page = await web_fetch(url)
        if not page.get("success"):
            raise ValueError(page.get("error") or "Fetch failed")
        return cls.add_text(
            title=page.get("title") or url, content=page.get("content", ""),
            source_type="url", uri=page.get("url") or url, tags=tags,
            metadata={"fetched_at": _now(), "length": page.get("length")},
            private=private, profile=profile, credibility=cls._credibility_for_url(url),
        )

    @classmethod
    def add_transcript(
        cls, title: str, content: str, people: Optional[List[str]] = None,
        source_type: str = "transcript", **kwargs,
    ) -> Dict[str, Any]:
        metadata = kwargs.pop("metadata", {}) or {}
        metadata["people"] = people or []
        return cls.add_text(
            title=title, content=content, source_type=source_type, metadata=metadata, **kwargs
        )

    # ------------------------------------------------------------------ search
    @classmethod
    def search(
        cls,
        query: str,
        *,
        limit: int = 10,
        source_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        tags: Optional[List[str]] = None,
        people: Optional[List[str]] = None,
        profile: Optional[str] = None,
        private: Optional[bool] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(limit, 50))
        where = []
        params: List[Any] = []
        if source_type:
            where.append("s.source_type = ?")
            params.append(source_type)
        if since:
            where.append("s.created_at >= ?")
            params.append(since)
        if until:
            where.append("s.created_at <= ?")
            params.append(until)
        if profile:
            where.append("s.profile = ?")
            params.append(profile)
        if private is True:
            where.append("s.private = 1")
        elif private is False:
            where.append("s.private = 0")
        if source_id:
            where.append("s.id = ?")
            params.append(source_id)
        where_sql = f"AND {' AND '.join(where)}" if where else ""

        fts_rows = []
        with db_session() as conn:
            fts_expr = _fts_query(query)
            if fts_expr:
                try:
                    fts_rows = conn.execute(
                        f"""
                        SELECT f.chunk_id, f.source_id, bm25(knowledge_fts) AS rank
                        FROM knowledge_fts f
                        JOIN knowledge_sources s ON s.id = f.source_id
                        WHERE knowledge_fts MATCH ? {where_sql}
                        ORDER BY rank ASC LIMIT 200
                        """,
                        [fts_expr, *params],
                    ).fetchall()
                except Exception:
                    fts_rows = []
            candidates = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.source_id, c.content, c.embedding, c.idx,
                       s.title, s.source_type, s.uri, s.author, s.tags, s.metadata,
                       s.created_at, s.credibility, s.private, s.profile
                FROM knowledge_chunks c JOIN knowledge_sources s ON s.id = c.source_id
                WHERE 1=1 {where_sql}
                ORDER BY c.id DESC LIMIT 5000
                """,
                params,
            ).fetchall()

        query_vec = embed_text(query)
        query_tokens = set(_tokens(query))
        fts_scores = {}
        if fts_rows:
            for row in fts_rows:
                rank = abs(float(row["rank"] or 0.0))
                fts_scores[row["chunk_id"]] = 1.0 / (1.0 + rank)

        scored = []
        for row in candidates:
            content = row["content"] or ""
            content_tokens = set(_tokens(content))
            lexical = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            try:
                vec = json.loads(row["embedding"] or "[]")
            except Exception:
                vec = []
            semantic = _cosine(query_vec, vec) if vec else 0.0
            keyword = max(lexical, fts_scores.get(row["chunk_id"], 0.0))
            metadata_score = 0.0
            title_l = (row["title"] or "").lower()
            if any(t in title_l for t in query_tokens):
                metadata_score += 0.6
            try:
                meta = json.loads(row["metadata"] or "{}")
            except Exception:
                meta = {}
            src_tags = []
            try:
                src_tags = json.loads(row["tags"] or "[]")
            except Exception:
                pass
            if tags and not set(t.lower() for t in tags) & set(str(t).lower() for t in src_tags):
                continue
            source_people = [str(p).lower() for p in (meta.get("people") or [])]
            if people and not set(p.lower() for p in people) & set(source_people + [str(row["author"] or "").lower()]):
                continue
            if source_people and people:
                metadata_score += 0.2
            score = 0.58 * keyword + 0.32 * semantic + 0.10 * min(1.0, metadata_score)
            if score <= 0.01:
                continue
            passage, highlight = cls._passage(content, query_tokens)
            scored.append({
                "chunk_id": row["chunk_id"],
                "source_id": row["source_id"],
                "title": row["title"],
                "source_type": row["source_type"],
                "uri": row["uri"],
                "author": row["author"],
                "tags": src_tags,
                "created_at": row["created_at"],
                "credibility": row["credibility"],
                "private": bool(row["private"]),
                "profile": row["profile"],
                "score": round(score, 4),
                "passage": passage,
                "highlight": highlight,
                "citation": cls._citation(row),
            })
        scored.sort(key=lambda r: (-r["score"], r["chunk_id"]))
        # de-duplicate adjacent chunks from the same source for readability
        seen_sources: Dict[str, int] = {}
        results = []
        for row in scored:
            seen_sources[row["source_id"]] = seen_sources.get(row["source_id"], 0) + 1
            if seen_sources[row["source_id"]] > 3:
                continue
            results.append(row)
            if len(results) >= limit:
                break
        return results

    @classmethod
    def ask(
        cls,
        question: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        synthesize: bool = False,
        answer_style: str = "concise",
    ) -> Dict[str, Any]:
        filters = filters or {}
        results = cls.search(question, limit=int(filters.pop("limit", 8)), **filters)
        if not results:
            return {
                "question": question,
                "answer": "I couldn't find anything in the local knowledge vault for that question.",
                "citations": [], "results": [], "grounded": False,
            }
        citations = []
        seen = set()
        for idx, result in enumerate(results, 1):
            result["citation_index"] = idx
            key = result["source_id"]
            if key not in seen:
                citations.append({
                    "index": idx, "source_id": result["source_id"],
                    "title": result["title"], "uri": result["uri"],
                    "source_type": result["source_type"], "credibility": result["credibility"],
                })
                seen.add(key)
        answer = None
        if synthesize:
            answer = cls._llm_answer(question, results, answer_style)
        if not answer:
            answer = cls._extractive_answer(question, results)
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "results": results,
            "grounded": True,
            "synthesized": bool(synthesize and answer),
        }

    @classmethod
    def _llm_answer(cls, question: str, results: List[Dict[str, Any]], answer_style: str) -> Optional[str]:
        try:
            import asyncio
            import concurrent.futures

            from services.models import ModelService

            sources = "\n\n".join(
                f"[{i}] {r['title']} ({r['uri'] or r['source_type']})\n{r['passage']}"
                for i, r in enumerate(results, 1)
            )
            messages = [
                {"role": "system", "content": (
                    "Answer only from the supplied local sources. Cite every factual claim with [n]. "
                    "If sources disagree, say so explicitly. If the answer is not present, say so. "
                    f"Style: {answer_style}."
                )},
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}"},
            ]

            def _call() -> Optional[str]:
                return asyncio.run(ModelService.complete(messages, max_tokens=900, profile="balanced"))

            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(_call).result(timeout=120)
            except RuntimeError:
                return _call()
        except Exception:
            return None

    @classmethod
    def _extractive_answer(cls, question: str, results: List[Dict[str, Any]]) -> str:
        q_tokens = set(_tokens(question))
        sentences = []
        for result in results[:5]:
            for sent in re.split(r"(?<=[.!?])\s+", result["passage"]):
                sent = sent.strip()
                if 30 <= len(sent) <= 500:
                    overlap = len(q_tokens & set(_tokens(sent)))
                    sentences.append((overlap, sent, result["citation_index"]))
        sentences.sort(key=lambda s: (-s[0], -len(s[1])))
        chosen = []
        used = set()
        for _, sent, idx in sentences:
            if sent in used:
                continue
            used.add(sent)
            chosen.append(f"{sent} [{idx}]")
            if len(chosen) >= 4:
                break
        if not chosen:
            return "Relevant local sources were found, but no extractive passage could be selected."
        return "Based on your local sources:\n\n" + "\n".join(f"- {c}" for c in chosen)

    # -------------------------------------------------------------------- CRUD
    @classmethod
    def list_sources(
        cls, *, source_type: Optional[str] = None, limit: int = 100, offset: int = 0,
        q: Optional[str] = None,
    ) -> Dict[str, Any]:
        where, params = [], []
        if source_type:
            where.append("source_type = ?")
            params.append(source_type)
        if q:
            where.append("(title LIKE ? OR uri LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        with db_session() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS n FROM knowledge_sources {where_sql}", params).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM knowledge_sources {where_sql} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                [*params, max(1, min(limit, 500)), max(0, offset)],
            ).fetchall()
            results = []
            for row in rows:
                item = cls._source_row(row)
                item["chunk_count"] = cls._chunk_count(row["id"])
                results.append(item)
        return {"total": total, "sources": results}

    @classmethod
    def get_source(cls, source_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM knowledge_sources WHERE id = ?", (source_id,)).fetchone()
            if not row:
                return None
            chunks = conn.execute(
                "SELECT id, idx, content, token_estimate FROM knowledge_chunks WHERE source_id = ? ORDER BY idx",
                (source_id,),
            ).fetchall()
        item = cls._source_row(row)
        item["chunks"] = [dict(c) for c in chunks]
        return item

    @classmethod
    def delete_source(cls, source_id: str) -> bool:
        with db_session() as conn:
            chunk_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM knowledge_chunks WHERE source_id = ?", (source_id,)
            ).fetchall()]
            for cid in chunk_ids:
                try:
                    conn.execute("DELETE FROM knowledge_fts WHERE chunk_id = ?", (cid,))
                except Exception:
                    pass
            conn.execute("DELETE FROM knowledge_chunks WHERE source_id = ?", (source_id,))
            cur = conn.execute("DELETE FROM knowledge_sources WHERE id = ?", (source_id,))
            deleted = cur.rowcount > 0
        if deleted:
            try:
                from services.audit import AuditService

                AuditService.log("knowledge", "knowledge.deleted", actor="user", summary=f"Deleted knowledge source {source_id}")
            except Exception:
                pass
        return deleted

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        with db_session() as conn:
            sources = conn.execute("SELECT COUNT(*) AS n FROM knowledge_sources").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) AS n FROM knowledge_chunks").fetchone()["n"]
            chars = conn.execute("SELECT COALESCE(SUM(char_count), 0) AS n FROM knowledge_sources").fetchone()["n"]
            embedded = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge_chunks WHERE embedding IS NOT NULL AND embedding != ''"
            ).fetchone()["n"]
            by_type = conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM knowledge_sources GROUP BY source_type"
            ).fetchall()
        return {
            "sources": sources, "chunks": chunks, "chars": chars,
            "embedded_chunks": embedded, "embedding_dim": EMBED_DIM,
            "by_type": {row["source_type"]: row["n"] for row in by_type},
        }

    @classmethod
    def reindex(cls) -> Dict[str, Any]:
        with db_session() as conn:
            try:
                conn.execute("DELETE FROM knowledge_fts")
            except Exception:
                pass
            rows = conn.execute(
                "SELECT c.id, c.source_id, c.content, s.title, s.tags FROM knowledge_chunks c "
                "JOIN knowledge_sources s ON s.id = c.source_id ORDER BY c.id"
            ).fetchall()
            for row in rows:
                try:
                    conn.execute(
                        "INSERT INTO knowledge_fts (chunk_id, source_id, title, content, tags) VALUES (?, ?, ?, ?, ?)",
                        (row["id"], row["source_id"], row["title"] or "", row["content"], row["tags"] or ""),
                    )
                except Exception:
                    pass
            missing = conn.execute(
                "SELECT id, content FROM knowledge_chunks WHERE embedding IS NULL OR embedding = '' LIMIT 2000"
            ).fetchall()
            for row in missing:
                conn.execute("UPDATE knowledge_chunks SET embedding = ? WHERE id = ?", (json.dumps(embed_text(row["content"])), row["id"]))
        return {"chunks": len(rows), "embeddings_repaired": len(missing)}

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _source_row(row) -> Dict[str, Any]:
        def _load(value, default):
            try:
                return json.loads(value) if value else default
            except Exception:
                return default

        return {
            "id": row["id"], "title": row["title"], "source_type": row["source_type"],
            "uri": row["uri"], "author": row["author"],
            "tags": _load(row["tags"], []), "metadata": _load(row["metadata"], {}),
            "content_hash": row["content_hash"], "char_count": row["char_count"],
            "private": bool(row["private"]), "profile": row["profile"],
            "credibility": row["credibility"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _chunk_count(source_id: str) -> int:
        with db_session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge_chunks WHERE source_id = ?", (source_id,)
            ).fetchone()
        return int(row["n"])

    @staticmethod
    def _citation(row) -> str:
        uri = f" — {row['uri']}" if row["uri"] else ""
        return f"{row['title']}{uri}"

    @staticmethod
    def _passage(content: str, query_tokens: set, width: int = 800) -> tuple:
        content = content.strip()
        if len(content) <= width:
            return content, content
        lower = content.lower()
        positions = [lower.find(tok) for tok in query_tokens if tok and lower.find(tok) >= 0]
        if not positions:
            return content[:width].strip(), content[:width].strip()
        pos = min(positions)
        start = max(0, pos - width // 3)
        passage = content[start:start + width].strip()
        return passage, passage

    @staticmethod
    def _credibility_for_url(url: str) -> float:
        host = (urlparse(url).hostname or "").lower() if url else ""
        high = (".gov", ".edu", "who.int", "nature.com", "science.org", "arxiv.org", "reuters.com", "apnews.com")
        low = ("facebook.com", "x.com", "twitter.com", "tiktok.com", "reddit.com")
        if host.endswith(high) or any(host.endswith(h) for h in high):
            return 0.9
        if any(host.endswith(h) for h in low):
            return 0.35
        return 0.6
