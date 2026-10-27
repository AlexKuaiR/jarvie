"""Cover letters and essays for specific postings, versioned, with unsupported claims flagged.

The writer (Opus 5.5) may research the company on the web, but everything it says
about you must come from your resume. It lists each claim it made about you; any
claim whose key words don't appear in your resume is flagged for you to check
before you use the letter. Nothing here submits anything anywhere.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from ..config import TZ
from ..llm import AgentModel, Usage, cached_system
from ..storage import dumps, loads
from . import postings
from .profile import CareerProfile
from .schema import connect

KINDS = ("cover_letter", "essay")


def _now() -> str:
    return datetime.now(TZ).isoformat()


def _decode(r) -> dict:
    out = dict(r)
    out["flagged"] = loads(out["flagged"], [])
    return out


def add_version(
    posting_id: int,
    kind: str,
    text: str,
    question: str | None = None,
    flagged: list[str] | None = None,
    research_notes: str | None = None,
    job_id: int | None = None,
) -> int:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not text.strip():
        raise ValueError("a document needs text")
    with connect() as conn:
        version = conn.execute(
            """SELECT COALESCE(MAX(version), 0) + 1 FROM documents
               WHERE posting_id = ? AND kind = ? AND COALESCE(question, '') = COALESCE(?, '')""",
            (posting_id, kind, question),
        ).fetchone()[0]
        return conn.execute(
            """INSERT INTO documents (posting_id, kind, question, version, text, flagged,
               research_notes, created_at, found_by_job) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                posting_id,
                kind,
                question,
                version,
                text.strip(),
                dumps(flagged or []),
                research_notes,
                _now(),
                job_id,
            ),
        ).lastrowid


def get(document_id: int) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _decode(r) if r else None


def for_posting(posting_id: int | None = None) -> list[dict]:
    """Documents, newest first, optionally for one posting (with its company and title)."""
    sql = """SELECT documents.*, postings.company, postings.title FROM documents
             JOIN postings ON postings.id = documents.posting_id"""
    args: tuple = ()
    if posting_id is not None:
        sql += " WHERE posting_id = ?"
        args = (posting_id,)
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY documents.id DESC", args).fetchall()
    return [_decode(r) for r in rows]


STOPWORDS = set(
    "a an the and or of to in on for with at by from my i me was is am were be been "
    "this that these those as it its into over under about during through".split()
)


def _tokens(text: str) -> list[str]:
    # Keep "c++", "c#", "node.js" whole, but drop sentence periods ("2025." -> "2025").
    return [t.strip(".") for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if t.strip(".")]


def unsupported(claims: list[str], resume: str) -> list[str]:
    """Claims whose substantive words mostly don't appear in the resume."""
    resume_words = set(_tokens(resume))
    flagged = []
    for claim in claims:
        words = [w for w in _tokens(claim) if w not in STOPWORDS]
        numbers = [w for w in words if re.fullmatch(r"\d[\d.,%]*", w)]
        if any(n not in resume_words for n in numbers):
            flagged.append(claim)  # a number not in the resume is the riskiest kind
            continue
        if words and sum(w in resume_words for w in words) / len(words) < 0.5:
            flagged.append(claim)
    return flagged


SYSTEM = """\
You write cover letters and application essays for one student, tailored to one
posting. Rules:
- Everything about the student comes from their resume below. Never invent
  experience, numbers, awards, coursework or skills.
- Facts about the company or lab may come from the posting or pages you read with
  web_search/web_fetch; prefer their own site. Keep research brief.
- Specific, warm, plain language; no clichés ("I am writing to express"), no flattery.
- Cover letters: 250-350 words, 3-4 paragraphs, addressed to the hiring team unless a
  name is known. Essays: answer the exact prompt within its word limit.
- When done, call save_document once with the text, every claim you made about the
  student (one short sentence each), and a few notes on what you learned about them.
"""


@dataclass
class DocLog:
    saved: list[int] = field(default_factory=list)


def write_document(job: dict, agent: AgentModel, profile: CareerProfile, spend: Usage) -> dict:
    p = job["params"]
    kind = p.get("kind", "cover_letter")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    posting = postings.get(int(p.get("posting_id") or 0))
    if posting is None:
        raise ValueError(f"no posting {p.get('posting_id')}")
    question = (p.get("question") or "").strip() or None
    if kind == "essay" and not question:
        raise ValueError("an essay needs the prompt (question)")
    word_limit = p.get("word_limit")
    chosen = profile.resume_for(posting)  # the swe/mle/qr resume this posting uses
    resume = chosen.text()
    log = DocLog()

    def save_document(text: str, claims_about_student: list[str], research_notes: str = "") -> str:
        """Save the finished document.

        Args:
            text: The full cover letter or essay, plain text.
            claims_about_student: Each claim made about the student, one sentence each.
            research_notes: Brief notes on what you learned about the company or lab.
        """
        if word_limit and len(text.split()) > int(word_limit):
            return f"error: {len(text.split())} words; the limit is {word_limit}. Shorten it."
        flagged = unsupported(claims_about_student, resume)
        doc_id = add_version(
            posting["id"], kind, text, question, flagged, research_notes, job["id"]
        )
        log.saved.append(doc_id)
        return f"saved document {doc_id}" + (f"; {len(flagged)} claims flagged" if flagged else "")

    stable = f"{SYSTEM}\n## Student\n{profile.name or 'the student'}\n## Resume\n{resume}"
    task = [
        f"Write a {'cover letter' if kind == 'cover_letter' else 'essay'} for: "
        f"{posting['title']} at {posting['company']} ({posting['url'] or 'no url'}).",
        f"Posting text:\n{(posting['raw_text'] or 'not saved; read the url if there is one')[:8000]}",
    ]
    if question:
        task.append(
            f"Essay prompt: {question}" + (f" (limit {word_limit} words)" if word_limit else "")
        )
    if p.get("notes"):
        task.append(f"The student's notes: {p['notes']}")
    result = agent.run(
        cached_system(stable),
        "\n\n".join(task),
        [save_document],
        web_search=5,
        web_fetch=3 if posting["url"] else 0,
    )
    spend.add(result.usage)
    if not log.saved:
        raise ValueError(f"the writer didn't save a document: {result.text[:300]}")
    doc = get(log.saved[-1])
    return {
        "document_id": doc["id"],
        "version": doc["version"],
        "flagged": doc["flagged"],
        "resume": chosen.name,
    }
