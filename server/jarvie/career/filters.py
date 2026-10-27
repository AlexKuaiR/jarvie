"""Hard filters and keyword tagging, applied before any model sees a posting.

These are cheap and deterministic, so they run on every fetched posting; only what
survives goes on to the (paid) research agent for scoring. Each rule only drops a
posting when the text clearly says so: an unknown pay or graduation date keeps it.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from .sources import RawPosting

# Field name -> keywords that suggest it. Matched as whole words, case-insensitive.
FIELD_KEYWORDS = {
    "machine learning": [
        "machine learning",
        "ml",
        "deep learning",
        "ai",
        "artificial intelligence",
        "nlp",
        "computer vision",
        "llm",
        "reinforcement learning",
        "data science",
    ],
    "software engineering": [
        "software",
        "swe",
        "backend",
        "back-end",
        "frontend",
        "front-end",
        "full stack",
        "full-stack",
        "developer",
        "infrastructure",
        "platform engineer",
        "mobile",
    ],
    "biology": [
        "biology",
        "biological",
        "molecular",
        "cell",
        "neuroscience",
        "genetics",
        "lab",
        "wet lab",
        "life science",
        "life sciences",
        "ecology",
        "microbiology",
    ],
    "biotech": [
        "biotech",
        "biotechnology",
        "genomics",
        "protein",
        "drug discovery",
        "pharma",
        "therapeutics",
        "bioinformatics",
        "computational biology",
        "crispr",
        "biomedical",
    ],
    "quant": [
        "quant",
        "quantitative research",
        "quantitative researcher",
        "quantitative trading",
        "quantitative trader",
        "quantitative developer",
        "quantitative analyst",
        "quantitative finance",
        "trading",
        "trader",
        "market making",
        "market maker",
        "alpha research",
        "systematic trading",
        "high-frequency",
        "hft",
    ],
}

INTERN_WORDS = re.compile(
    r"\b(intern|internship|co-?op|summer (?:student|analyst|associate)|student researcher"
    r"|research assistant|undergraduate research|reu|fellowship)\b",
    re.I,
)
RESEARCH_WORDS = re.compile(
    r"\b(research assistant|student researcher|undergraduate research|reu|lab|research intern)\b",
    re.I,
)
UNPAID_WORDS = re.compile(
    r"\b(unpaid|volunteer|for (?:academic )?credit only|no compensation)\b", re.I
)
# The rest of the sentence after "graduate/graduating/graduation", where the years are.
GRAD_CLAUSE = re.compile(r"graduat\w*([^.\n]{0,100})", re.I)
YEAR = re.compile(r"\b(20\d\d)\b")


def _has_word(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", text, re.I) is not None


def tag_fields(title: str, text: str) -> list[str]:
    """Fields a posting looks like, from its title first and then its description."""
    haystack = f"{title}\n{text[:4000]}"
    return [f for f, words in FIELD_KEYWORDS.items() if any(_has_word(haystack, w) for w in words)]


COMPANY_SUFFIXES = re.compile(r"\b(inc|llc|ltd|corp|corporation|co|company|plc)\b")


def company_key(name: str) -> str:
    """A company name reduced for matching: "Johnson & Johnson" == "johnson and johnson",
    "Eli Lilly and Company" == "Eli Lilly"."""
    text = name.lower().replace("&", " and ")
    text = COMPANY_SUFFIXES.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", text))
    words = text.split()
    while words and words[-1] == "and":  # "Eli Lilly and Company" -> "eli lilly"
        words.pop()
    return " ".join(words)


def same_company(name: str, wanted: list[str]) -> bool:
    key = company_key(name)
    return bool(key) and any(key == company_key(w) for w in wanted)


def kind_of(title: str, text: str) -> str:
    return "research" if RESEARCH_WORDS.search(title) else "internship"


def grad_years(text: str) -> list[int]:
    """Every year mentioned in a sentence about graduating, e.g. "Dec 2027 or 2028"."""
    return sorted({int(y) for clause in GRAD_CLAUSE.findall(text) for y in YEAR.findall(clause)})


def _shift_months(d: date, months: int) -> date:
    total = d.year * 12 + d.month - 1 + months
    return date(total // 12, total % 12 + 1, 1)


@dataclass
class Criteria:
    """What one search wants: the profile's settings, narrowed by a brief's answers."""

    graduation: date
    window_months: int = 6
    fields: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=lambda: ["internship", "research"])
    locations: list[str] = field(default_factory=list)
    paid_only: bool = True
    term: str = ""  # e.g. "Summer 2027"; empty means any
    include_companies: list[str] = field(default_factory=list)
    exclude_companies: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    keep: bool
    reason: str = ""  # why it was dropped
    fields: list[str] = field(default_factory=list)
    kind: str = "internship"
    paid: bool | None = None


def judge(p: RawPosting, c: Criteria) -> Verdict:
    """Keep or drop one posting, with the reason when dropped."""
    text = f"{p.title}\n{p.location}\n{p.text}"
    fields = tag_fields(p.title, p.text)
    kind = kind_of(p.title, p.text)
    paid = False if UNPAID_WORDS.search(text) else None

    def drop(reason: str) -> Verdict:
        return Verdict(False, reason, fields, kind, paid)

    if same_company(p.company, c.exclude_companies):
        return drop("excluded company")
    wanted = same_company(p.company, c.include_companies)
    if not INTERN_WORDS.search(p.title) and not (p.terms and p.source == "simplify"):
        return drop("not an internship or research position")
    if kind not in c.kinds:
        return drop(f"{kind} not wanted")
    if c.fields and not wanted and not set(fields) & set(c.fields):
        return drop("field doesn't match")
    if c.paid_only and paid is False:
        return drop("unpaid")
    if c.term and p.terms and not any(c.term.lower() == t.lower() for t in p.terms):
        return drop("different term")
    if c.locations and p.location:
        loc = p.location.lower()
        if not any(x.lower() in loc for x in c.locations) and "remote" not in loc:
            return drop("location")
    years = grad_years(p.text)
    if years:
        # Year-level check: a May 2028 graduation with a 6-month window accepts 2027-2028.
        lo = _shift_months(c.graduation, -c.window_months).year
        hi = _shift_months(c.graduation, c.window_months).year
        if not any(lo <= y <= hi for y in years):
            return drop(f"graduation year {years} outside your window")
    return Verdict(True, "", fields, kind, paid)


def relevance(p: RawPosting, v: Verdict, c: Criteria) -> float:
    """A cheap ranking for postings that passed, so a capped search keeps the strongest.

    Field words in the title count more than in the description, an explicit match on
    the term asked for counts most, and companies you asked for come first.
    """
    score = 0.0
    wanted_fields = set(c.fields) or set(FIELD_KEYWORDS)
    title_fields = set(tag_fields(p.title, ""))
    score += 2.0 * len(title_fields & wanted_fields)
    score += 0.5 * len((set(v.fields) - title_fields) & wanted_fields)
    if c.term and any(c.term.lower() == t.lower() for t in p.terms):
        score += 3.0
    if same_company(p.company, c.include_companies):
        score += 5.0
    if p.ats in ("greenhouse", "lever", "ashby"):
        score += 0.5  # easiest to read and, later, to fill in
    return score
