import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .config import TZ
from .db import connect


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    url: str | None = None
    source: str = "manual"
    notes: str | None = None
    added_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> dict:
        """Short form for search results, so the LLM isn't handed every abstract at once."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
        }


def _row_to_paper(r) -> Paper:
    return Paper(
        paper_id=r["paper_id"],
        title=r["title"],
        authors=json.loads(r["authors"]) if r["authors"] else [],
        year=r["year"],
        venue=r["venue"],
        abstract=r["abstract"],
        doi=r["doi"],
        url=r["url"],
        source=r["source"],
        notes=r["notes"],
        added_at=r["added_at"],
    )


def normalize_doi(doi: str | None) -> str | None:
    # "https://doi.org/10.1/ABC" and "10.1/abc" are the same paper; DOIs are case-insensitive.
    if not doi or not doi.strip():
        return None
    doi = doi.strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)


def add(
    title: str,
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str | None = None,
    abstract: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    source: str = "manual",
    paper_id: str | None = None,
) -> str:
    """Store a paper and return its id.

    If a paper with the same DOI or paper_id is already stored, nothing is
    added and the existing id is returned, so caching a search result twice
    keeps the user's notes. paper_id is for sources with their own ids (S2);
    hand-entered papers get "manual:<n>".
    """
    if not title or not title.strip():
        raise ValueError("a paper needs a title")
    doi = normalize_doi(doi)
    with connect() as conn:
        if doi:
            existing = conn.execute("SELECT paper_id FROM papers WHERE doi = ?", (doi,)).fetchone()
            if existing:
                return existing["paper_id"]
        if paper_id is not None:
            seen = conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
            if seen:
                return paper_id
        else:
            next_n = conn.execute("SELECT COALESCE(MAX(rowid), 0) + 1 FROM papers").fetchone()[0]
            paper_id = f"manual:{next_n}"
        conn.execute(
            """INSERT INTO papers
               (paper_id, title, authors, year, venue, abstract, doi, url, source, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper_id,
                title.strip(),
                json.dumps(authors or []),
                year,
                venue,
                abstract,
                doi,
                url,
                source,
                datetime.now(TZ).isoformat(),
            ),
        )
        return paper_id


def get(paper_id: str) -> Paper | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _row_to_paper(r) if r else None


def by_doi(doi: str) -> Paper | None:
    doi = normalize_doi(doi)
    if not doi:
        return None
    with connect() as conn:
        r = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
    return _row_to_paper(r) if r else None


def search(query: str, limit: int = 5) -> list[Paper]:
    """Keyword search over title, abstract and authors, best matches first."""
    # Quote each word so FTS5 doesn't treat "-" or ":" in the query as syntax, and OR
    # them so a spoken query like "papers on sleep spindles" still matches.
    words = re.findall(r"\w+", query)
    if not words:
        return []
    match = " OR ".join(f'"{w}"' for w in words)
    sql = """SELECT papers.* FROM papers_fts
             JOIN papers ON papers.rowid = papers_fts.rowid
             WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?"""
    with connect() as conn:
        rows = conn.execute(sql, (match, limit)).fetchall()
    return [_row_to_paper(r) for r in rows]


def list_recent(limit: int = 20) -> list[Paper]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM papers ORDER BY added_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_paper(r) for r in rows]


def update_notes(paper_id: str, text: str) -> bool:
    with connect() as conn:
        cur = conn.execute("UPDATE papers SET notes = ? WHERE paper_id = ?", (text, paper_id))
        return cur.rowcount > 0


def delete(paper_id: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,)).rowcount > 0
