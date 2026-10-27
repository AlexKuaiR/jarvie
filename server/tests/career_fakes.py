"""Shared fakes for career/outreach tests: canned job-board JSON and a tool-call helper."""

import asyncio

import httpx

GREENHOUSE = {
    "meta": {"name": "Acme Bio"},
    "jobs": [
        {
            "title": "Machine Learning Intern, Summer 2027",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "location": {"name": "South San Francisco, CA"},
            "content": "&lt;p&gt;Train protein models. Graduating 2028.&lt;/p&gt;",
        },
        {
            "title": "Senior Accountant",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/2",
            "location": {"name": "Remote"},
            "content": "",
        },
        {
            "title": "Biology Research Intern",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/3",
            "location": {"name": "Boston, MA"},
            "content": "&lt;p&gt;Unpaid, for credit only. Wet lab.&lt;/p&gt;",
        },
    ],
}
LEVER = [
    {
        "text": "Software Engineering Intern",
        "hostedUrl": "https://jobs.lever.co/beta/1",
        "categories": {"location": "Berkeley, CA"},
        "descriptionPlain": "Backend services in Python. Expected graduation 2025.",
        "lists": [{"text": "Requirements", "content": "<li>Python</li>"}],
    }
]
ASHBY = {
    "jobs": [
        {
            "title": "AI Research Intern",
            "jobUrl": "https://jobs.ashbyhq.com/gamma/1",
            "location": "Remote",
            "descriptionPlain": "Deep learning research. Paid $50/hr.",
            "isListed": True,
        },
        {"title": "Hidden", "jobUrl": "https://jobs.ashbyhq.com/gamma/2", "isListed": False},
    ]
}
SIMPLIFY = [
    {
        "company_name": "Delta Therapeutics",
        "title": "Computational Biology Intern",
        "url": "https://delta.wd1.myworkdayjobs.com/x",
        "locations": ["Cambridge, MA"],
        "terms": ["Summer 2027"],
        "active": True,
        "is_visible": True,
        "category": "Data Science, AI & Machine Learning",
    },
    {
        "company_name": "Old Co",
        "title": "SWE Intern",
        "url": "https://x",
        "terms": ["Summer 2026"],
        "active": False,
    },
]


def board_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "greenhouse.io/v1/boards/acme" in url:
        return httpx.Response(200, json=GREENHOUSE)
    if "api.lever.co/v0/postings/beta" in url:
        return httpx.Response(200, json=LEVER)
    if "ashbyhq.com/posting-api/job-board/gamma" in url:
        return httpx.Response(200, json=ASHBY)
    if "SimplifyJobs/Summer2027-Internships" in url:
        return httpx.Response(200, json=SIMPLIFY)
    return httpx.Response(404, json={"error": "not found"})


def board_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(board_handler))


WATCHLIST = {
    "greenhouse": ["acme"],
    "lever": ["beta"],
    "ashby": ["gamma"],
    "simplify": ["SimplifyJobs/Summer2027-Internships"],
}


def write_profile(tmp_path, watchlist: dict | None = None) -> "object":
    """Write a career profile + resume under tmp_path and return the profile path."""
    resume = tmp_path / "resume.md"
    resume.write_text(
        "# Alex K\nUC Berkeley, B.S. EECS, expected May 2028.\n"
        "Skills: Python, PyTorch, molecular biology lab work.\n"
        "Projects: protein structure model; voice assistant (Pipecat)."
    )
    wl = watchlist if watchlist is not None else WATCHLIST
    lines = "\n".join(f"{k} = {list(v)!r}".replace("'", '"') for k, v in wl.items())
    path = tmp_path / "career_profile.toml"
    path.write_text(
        f'resume = "{resume}"\ngraduation = "2028-05"\n'
        'fields = ["machine learning", "software engineering", "biology", "biotech"]\n'
        "paid_only = true\nlocations = []\n"
        f"[watchlist]\n{lines}\n"
    )
    return path


class FakeParams:
    """Stands in for Pipecat's FunctionCallParams: records what the tool reports back."""

    def __init__(self):
        self.result = None

    async def result_callback(self, result):
        self.result = result


def call(tool, **kwargs) -> dict:
    params = FakeParams()
    asyncio.run(tool(params, **kwargs))
    return params.result
