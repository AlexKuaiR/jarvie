"""Questions Jarvie asks before a big search, answered by voice or by typing.

A brief is one search-to-be: its kind, the answers so far, and its status. Answers
live in the database, not in a conversation, so every channel writes to the same
brief: you can start answering by voice, finish in the localhost text box (or the
dashboard form), and submit from either. Submitting checks the required questions are
answered and only then puts the job on the queue, so no expensive run starts on a
half-specified request.
"""

import re
from dataclasses import dataclass
from datetime import datetime

from . import jobs
from .config import TZ
from .storage import connect_with, dumps, loads

BRIEFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS briefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,             -- the job kind it will start
    answers TEXT NOT NULL,          -- JSON dict, question key -> answer
    status TEXT NOT NULL DEFAULT 'open',   -- open, submitted, cancelled
    job_id INTEGER,
    last_channel TEXT,              -- 'voice', 'text' or 'dashboard', for the record
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

CHANNELS = ("voice", "text", "dashboard")


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str  # how Jarvie asks it, spoken or shown
    type: str  # 'text', 'int', 'bool', 'choice' (one of), 'choices' (some of), 'list', 'names'
    required: bool = True
    choices: tuple[str, ...] = ()
    default: object = None  # used when an optional question is left unanswered
    min: int | None = None
    max: int | None = None
    words: tuple[tuple[str, int], ...] = ()  # spoken words an 'int' accepts, e.g. ("all", -1)


# Question sets, one per job kind that runs a mass search. Kinds that act on one
# thing (score one posting, find one person's email) take their parameters directly.
QUESTIONS: dict[str, tuple[Question, ...]] = {
    "discover_postings": (
        Question(
            "kinds",
            "Internships, research positions, or both?",
            "choices",
            choices=("internship", "research"),
        ),
        Question(
            "fields",
            "Which fields this time? For example machine learning, software "
            "engineering, quant, biology, biotech.",
            "list",
        ),
        Question("term", "Which term are you looking for, like Summer 2027?", "text"),
        Question(
            "locations",
            "Any locations, or remote? Say none for anywhere.",
            "list",
            required=False,
            default=[],
        ),
        Question("paid_only", "Paid positions only?", "bool", required=False, default=True),
        Question(
            "min_prestige",
            "Minimum prestige, 1 to 5? 1 means anything.",
            "int",
            required=False,
            default=1,
            min=1,
            max=5,
        ),
        Question(
            "include_companies",
            "Any companies or labs to put first? Names only; they're searched if they're "
            "on your watchlist, or on the web when you search beyond the boards.",
            "names",
            required=False,
            default=[],
        ),
        Question("exclude_companies", "Any to skip?", "names", required=False, default=[]),
        Question(
            "max_results",
            "About how many results do you want?",
            "int",
            required=False,
            default=25,
            min=1,
            max=200,
        ),
        # The two AI gates: a search with both off uses no model at all (free, no key).
        Question(
            "search_beyond_boards",
            "Should I also search beyond the job boards, like lab pages, research "
            "programs and company sites? That uses AI.",
            "bool",
        ),
        Question(
            "ai_score",
            "How many of the results should AI score for fit and prestige? Say none, "
            "a number, or all. The rest are saved unscored; you can score any of them later.",
            "int",
            min=0,
            max=200,
            words=(("none", 0), ("no", 0), ("zero", 0), ("all", -1), ("every", -1)),
        ),
        Question(
            "notes",
            "Anything else I should know for this search?",
            "text",
            required=False,
            default="",
        ),
    ),
    "find_contacts": (
        Question(
            "target",
            "Which company, lab, school or organization? Say any for no specific one.",
            "text",
        ),
        Question(
            "roles",
            "What kind of people? For example university recruiter, admissions, "
            "ML engineer, professor.",
            "list",
        ),
        Question(
            "schools",
            "Only alumni of Berkeley, Newton North, either, or no preference?",
            "choices",
            required=False,
            default=[],
            choices=("berkeley", "nnhs"),
        ),
        Question(
            "purpose",
            "What's the outreach for: a coffee chat, a research inquiry, a "
            "referral, advice, or something else?",
            "choice",
            choices=("coffee_chat", "research_inquiry", "referral", "advice", "other"),
        ),
        Question(
            "max_contacts",
            "How many people at most?",
            "int",
            required=False,
            default=10,
            min=1,
            max=100,
        ),
        Question(
            "draft_after",
            "Should I draft emails to them once they're found?",
            "bool",
            required=False,
            default=False,
        ),
        Question(
            "resume",
            "Which of your resumes should the emails draw on? Say default if it doesn't matter.",
            "text",
            required=False,
            default="",
        ),
        Question(
            "notes",
            "Anything else, like a team or topic to focus on?",
            "text",
            required=False,
            default="",
        ),
    ),
}


class BriefError(Exception):
    """A bad answer or an action the brief's state doesn't allow; the message says why."""


def _now() -> str:
    return datetime.now(TZ).isoformat()


def connect():
    return connect_with(BRIEFS_SCHEMA)


def _questions(kind: str) -> dict[str, Question]:
    if kind not in QUESTIONS:
        raise BriefError(f"no questions for {kind}; kinds with briefs: {sorted(QUESTIONS)}")
    return {q.key: q for q in QUESTIONS[kind]}


def _split_list(value) -> list[str]:
    # Spoken or typed lists arrive as one string ("ML, biotech and biology") or a list.
    if isinstance(value, str):
        parts = value.replace(" and ", ",").split(",")
    else:
        parts = list(value)
    items = [str(p).strip() for p in parts if str(p).strip()]
    return [] if [i.lower() for i in items] in (["none"], ["any"], ["anywhere"]) else items


# Words that make a "company" answer a description ("promising start ups") rather than a name.
DESCRIPTION = re.compile(
    r"\b(start[ -]?ups?|companies|company|firms?|big|large|small|promising|top|any|labs)\b", re.I
)


def company_names(value) -> tuple[list[str], list[str]]:
    """Company names from an answer, plus any descriptive phrases (which belong in notes).

    A list is taken as given. A string is split on commas, semicolons, "or", "/" and
    "and", except that "Johnson and Johnson" stays whole (and "&" is never split).
    "Procter and Gamble" would split, which is why the voice model passes lists.
    "Big companies like Eli Lilly" gives the name "Eli Lilly" and keeps the phrase.
    """
    if isinstance(value, str):
        parts = []
        for chunk in re.split(r",|;|/|\bor\b", value):
            pieces = [x.strip() for x in re.split(r"\band\b", chunk)]
            # Rejoin "Johnson" + "Johnson" -> "Johnson and Johnson"; "&" is never split.
            for piece in pieces:
                if parts and piece and parts[-1].split()[-1:] == [piece] and piece[:1].isupper():
                    parts[-1] = f"{parts[-1]} and {piece}"
                else:
                    parts.append(piece)
    else:
        parts = [str(v) for v in value]
    names, phrases = [], []
    for part in (p.strip(" .") for p in parts):
        if not part or part.lower() in ("none", "any", "no"):
            continue
        if re.search(r"\b(like|such as|e\.g\.?)\s", part, re.I):
            phrases.append(part)
            part = re.split(r"\b(?:like|such as|e\.g\.?)\s", part, flags=re.I)[-1].strip(" .")
        if DESCRIPTION.search(part):
            phrases.append(part)
        elif part:
            names.append(part)
    return list(dict.fromkeys(names)), list(dict.fromkeys(phrases))


def _coerce(q: Question, value):
    """Turn a spoken or typed answer into the stored form, or raise BriefError."""
    if q.type == "text":
        return str(value).strip()
    if q.type == "list":
        return _split_list(value)
    if q.type == "names":
        return company_names(value)[0]
    if q.type == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("yes", "y", "true", "yeah", "yep", "sure"):
            return True
        if text in ("no", "n", "false", "nope"):
            return False
        raise BriefError(f"{q.key}: answer yes or no")
    if q.type == "int":
        spoken = dict(q.words)
        if isinstance(value, str) and value.strip().lower() in spoken:
            return spoken[value.strip().lower()]
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise BriefError(f"{q.key}: answer with a number") from None
        if (q.min is not None and number < q.min) or (q.max is not None and number > q.max):
            raise BriefError(f"{q.key}: answer between {q.min} and {q.max}")
        return number
    if q.type in ("choice", "choices"):
        picked = _split_list(value)
        if q.type == "choices" and [p.lower() for p in picked] in (["both"], ["either"]):
            picked = list(q.choices)
        picked = [p.lower().replace(" ", "_").replace("-", "_") for p in picked]
        if q.type == "choices" and not q.required and picked in ([], ["no_preference"]):
            return []  # "any" / "no preference" on an optional multi-choice
        bad = [p for p in picked if p not in q.choices]
        if bad or not picked:
            raise BriefError(f"{q.key}: choose from {', '.join(q.choices)}")
        if q.type == "choice":
            if len(picked) != 1:
                raise BriefError(f"{q.key}: choose just one of {', '.join(q.choices)}")
            return picked[0]
        return list(dict.fromkeys(picked))  # drop repeats, keep order
    raise ValueError(f"unknown question type {q.type}")


def _decode(r) -> dict:
    out = dict(r)
    out["answers"] = loads(out["answers"], {})
    return out


def start(kind: str, prefill: dict | None = None, channel: str = "voice") -> int:
    """Open a brief. prefill holds answers already known (e.g. from your profile), which
    still count as answered but can be changed before submitting."""
    questions = _questions(kind)
    answers = {k: _coerce(questions[k], v) for k, v in (prefill or {}).items() if k in questions}
    now = _now()
    with connect() as conn:
        return conn.execute(
            """INSERT INTO briefs (kind, answers, last_channel, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (kind, dumps(answers), channel, now, now),
        ).lastrowid


def get(brief_id: int) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM briefs WHERE id = ?", (brief_id,)).fetchone()
    return _decode(r) if r else None


def _open(brief_id: int) -> dict:
    brief = get(brief_id)
    if brief is None:
        raise BriefError(f"no brief {brief_id}")
    if brief["status"] != "open":
        raise BriefError(f"brief {brief_id} is already {brief['status']}")
    return brief


def remaining(brief_id: int) -> list[dict]:
    """Unanswered questions, required ones first, ready to be asked aloud or shown."""
    brief = _open(brief_id)
    unanswered = [q for q in QUESTIONS[brief["kind"]] if q.key not in brief["answers"]]
    unanswered.sort(key=lambda q: not q.required)
    return [
        {
            "key": q.key,
            "prompt": q.prompt,
            "required": q.required,
            "choices": list(q.choices),
            "default": q.default,
        }
        for q in unanswered
    ]


def answer(brief_id: int, answers: dict, channel: str = "voice") -> dict:
    """Record one or more answers. Every answer is checked before any is saved, so a bad
    one leaves the brief unchanged. Returns the updated brief."""
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    brief = _open(brief_id)
    questions = _questions(brief["kind"])
    unknown = set(answers) - set(questions)
    if unknown:
        raise BriefError(f"no such questions: {sorted(unknown)}")
    coerced = {k: _coerce(questions[k], v) for k, v in answers.items()}
    merged = {**brief["answers"], **coerced}
    # Descriptions given as companies ("promising start ups") go to notes, not lost.
    phrases = [
        phrase
        for k, v in answers.items()
        if questions[k].type == "names"
        for phrase in company_names(v)[1]
    ]
    if phrases and "notes" in questions:
        merged["notes"] = "; ".join(filter(None, [merged.get("notes", ""), *phrases]))
    with connect() as conn:
        conn.execute(
            "UPDATE briefs SET answers = ?, last_channel = ?, updated_at = ? WHERE id = ?",
            (dumps(merged), channel, _now(), brief_id),
        )
    return get(brief_id)


def submit(brief_id: int, channel: str = "voice") -> int:
    """Start the search: fill defaults for skipped optional questions, queue the job,
    and return its id. Refused while any required question is unanswered."""
    brief = _open(brief_id)
    missing = [
        q.key for q in QUESTIONS[brief["kind"]] if q.required and q.key not in brief["answers"]
    ]
    if missing:
        raise BriefError(f"still need: {', '.join(missing)}")
    params = {q.key: brief["answers"].get(q.key, q.default) for q in QUESTIONS[brief["kind"]]}
    params["brief_id"] = brief_id
    job_id = jobs.enqueue(brief["kind"], params)
    with connect() as conn:
        conn.execute(
            """UPDATE briefs SET status = 'submitted', job_id = ?, last_channel = ?,
               updated_at = ? WHERE id = ?""",
            (job_id, channel, _now(), brief_id),
        )
    return job_id


def cancel(brief_id: int) -> None:
    _open(brief_id)
    with connect() as conn:
        conn.execute(
            "UPDATE briefs SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (_now(), brief_id),
        )


def open_briefs() -> list[dict]:
    """Briefs still being answered, newest first, so any channel can pick one up."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM briefs WHERE status = 'open' ORDER BY id DESC"
        ).fetchall()
    return [_decode(r) for r in rows]
