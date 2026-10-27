# Career pipeline and outreach: build log

Branch `career-pipeline`, cut from `main`. Every phase is its own commit, with tests:

| Commit | What it added |
|---|---|
| `e30348d` | Phase 0: postings and application tracking, standalone outreach store, job queue, briefs |
| `e920bbf` | Phase 1: job-board feeds, hard filters, dashboard, voice/text tools |
| `71a8f65` | Phase 2: Claude research agent and background worker |
| `2d27fc7` | Phase 3: people finder and Hunter email lookup |
| `79e82e9` | Phase 4: email writer, approval queue, scheduled sending, reply tracking |
| `d90d3dd` | Phase 5: cover letters and essays |
| `e3431e5` | Phase 6: apply assistant (fills forms, never submits) |
| `874c2a7`, `b6586e2` | Fixes found by smoke tests: clear missing-key error, tool-count test |
| (CLAUDE.md commit) | CLAUDE.md describes the new packages |

Nothing is pushed.

---

## 1. Architecture

```
 you ── voice ──┐                         you ── browser: http://localhost:7861 (dashboard)
 you ── text box (localhost:7860/client) ─┤                   │ review, edit, approve, set status
                ▼                         ▼                   ▼
   bot.py  (OpenAI gpt-4.1, unchanged)    31 tools ──►  SQLite ~/jarvie-data/jarvie.db  ◄── dashboard.py
     ├─ job_tools.py   briefs + job status          postings · applications · status_events
     ├─ career/tools.py postings, status, docs      documents · contacts · outreach · jobs · briefs
     └─ outreach/tools.py people, email, drafts           ▲
                                                          │ jobs table = queue between processes
                                worker.py  (separate process: python -m jarvie.worker)
          ┌───────────────────────┬────────────────────┬───────────────────────┬──────────────────┐
  career/research_agent.py   career/sources.py     outreach/research.py   outreach/writer.py   outreach/sender.py
  Opus 5.5 + web search/     Greenhouse, Lever,    Opus 5.5 people        Sonnet 5.5 drafts +  schedule, send
  fetch, rubric scoring      Ashby, SimplifyJobs   finder (no LinkedIn)   grounding check,     (dry run / Gmail),
  career/documents.py        (public JSON, free)   + Hunter email lookup  Batches API ≥10      replies, bounces
  Opus 5.5 letters/essays                                                                     
  career/apply_assist.py: visible Playwright browser, fills forms, never submits (run by hand)
```

- **Three models, three jobs.**
  - The voice model (OpenAI, `bot.py`) only talks to you, reads the database and queues jobs.
  - The research agent (Claude Opus 5.5) does all web work.
  - The writer (Claude Sonnet 5.5) writes emails from the facts the agent stored about each
    person.
  - All Claude calls go through `jarvie/llm.py`. It uses adaptive thinking, the server-side
    refusal fallback, cached system prompts (your resume and the rubric) and a cost estimate
    per call.
- **The worker is a separate process.** Model calls take minutes; the voice pipeline must stay
  responsive. The bot and dashboard only add rows to the `jobs` table and read results back.
- **Briefs hold the questions asked before a mass search.** The answers live in the database,
  so voice, the text box and the dashboard all edit the same brief. Submitting is refused until
  the required answers are in.
- **Outreach doesn't depend on the career package.**
  - A contact needs no company or posting.
  - An email has a purpose (coffee chat, research inquiry, referral, advice, other) and can
    optionally link to a posting.
  - The only coupling is that the writer signs with your name and uses your resume from the
    career profile.
- **Everything external sits behind an interface with a fake for tests:** `PostingSource`,
  `AgentModel`, `WriterModel`, `EmailFinder`, `Mailer`, `Browser`.

## 2. Features

| Area | What works | How you use it |
|---|---|---|
| Profile | Graduation month and window, fields, kinds, paid only, locations, minimum prestige, dealbreakers, watchlist, `[applicant]` details | `~/jarvie-data/career_profile.toml` (copy the example) |
| Job boards | Greenhouse, Lever and Ashby board APIs plus SimplifyJobs lists; free, no key | `python -m jarvie.career.ingest`, or part of every search |
| Hard filters | Internship or research, field (keyword tags), unpaid, term, location, graduation year window, include/exclude companies; a cheap relevance rank so capped searches keep the best | Automatic |
| Handshake | Paste a link or the posting text; the agent reads and scores it. Never scraped | "Add this posting: …" (voice or text box) |
| Scoring | Prestige, competitiveness and ownership of work (1–5 each, with reasons), fit score 0–1 with reasons, skills, application steps, deadline, pay | Automatic in searches |
| Searches | Brief → job boards (free) → two AI gates you answer each time: **search beyond the boards?** (AI web search for company pages, research programs, REUs, labs) and **how many results should AI score?** (none, a number, or all; the top-ranked go first). Answering no and none gives a free search that never loads a model, needs no key, and still runs after the daily AI budget is spent | "Find me internships", answered by voice, text or dashboard |
| Score on demand | Score any saved posting later, only the ones you care about | "Score the CACI one", or "Score with AI" on a posting page |
| Tracking | Interested, applied, OA, interviewing, offer, rejected, withdrawn; history with notes; coffee-chat flag; pipeline summary | Voice ("I got an OA from Acme") or the dashboard |
| People finder | Public pages only, LinkedIn refused, an evidence URL required for every person and every alumni claim, capped per job | Brief kind `find_contacts`: "find Berkeley alumni recruiters at Genentech" |
| Email lookup | Hunter finder and verifier; addresses that verify invalid are dropped; works without the career package | "Find the email for Dana Lee at gene.com", or the button on the Contacts page |
| Your own contacts | Add people yourself (no evidence needed) | "Add Professor Kim, kim@berkeley.edu", or the Contacts page form |
| Email drafts | Sonnet writes 90–140 words from stored facts. A code check rejects invented facts, placeholders, a missing name or the wrong length; one retry, otherwise rejected. 10 or more drafts go through the Batches API at half price | "Draft coffee chat emails to them", or "draft after" in a people search |
| Approval | Edit, approve, approve all, reject, cancel. Sending can't skip approval; one active thread per person (enforced by the database); do-not-contact stops queued mail | Dashboard → Outreach |
| Sending | Weekdays 9–5 (your time zone), 20–60 minutes apart, at most 15 a day. Dry run by default; Gmail when enabled | Automatic in the worker |
| Replies | Gmail thread checked every 10 minutes → replied or bounced; mark "had coffee chat" from a reply | Dashboard → Outreach |
| Cover letters and essays | Opus with company research, versioned per posting. Claims about you that aren't in your resume are flagged. Essay word limits are enforced | "Write a cover letter for the Acme one", or "Write with Claude" on a posting |
| Apply assistant | Greenhouse, Lever and Ashby forms: fills your details, uploads your resume and the latest cover letter, takes a screenshot and stops. EEO/demographic questions, checkboxes and custom questions are left for you. Workday and Handshake are refused | `python -m jarvie.career.apply_assist <posting_id>` |
| Several resumes | Named resumes (e.g. swe, mle, qr), each with a text copy, a PDF and the fields it's for. Each posting uses the best match by fields (title first), or the one you pick. Scoring sees all of them and records which it judged against; cover letters, the claim check and the apply assistant's upload use the posting's resume; emails use the one you name, a linked posting's, or the default. Quant roles are tagged as field `quant` | Posting page → Resume, or "use my QR resume for the Citadel one" |
| Cost control | Per-job budget (`CAREER_JOB_BUDGET_USD`, default $3), daily budget (`JARVIE_DAILY_BUDGET_USD`, default $10), cost logged per job | Dashboard → Jobs |

## 3. Test results

- **Branch suite:** 208 passed, 1 skipped, 0 failed
  (`server/.venv/bin/python -m pytest server -q`). The skip is the real-browser test, which runs
  once Chromium is installed. Lint is clean.
- **New tests:**
  - `test_career_profile.py`, `test_career_postings.py`, `test_outreach.py`, `test_jobs.py`,
    `test_briefs.py` (phase 0)
  - `test_career_phase1.py`, `test_career_phase2.py`, `test_outreach_phase3.py`,
    `test_outreach_phase4.py`, `test_career_phase5.py`, `test_career_phase6.py`
  - `test_career_resumes.py` (several resumes)
  - Fakes in `tests/career_fakes.py` and `tests/agent_fakes.py`
- **Trial merge of `career-pipeline` + `semantic-scholar` + `wakeword-eval` into `main`:**
  327 passed, 3 skipped. No files are deleted relative to any branch. Two files need a manual
  merge; see section 6.
- **Live smoke tests, no keys needed:**
  - Real feeds: Anthropic on Greenhouse returned 679 postings, Palantir on Lever 307, OpenAI on
    Ashby 796, Simplify about 4,900.
  - A real ingest saved 1,655 postings from 6,366 fetched, with no errors.
  - The real dashboard server returned 200 on every page.
- **Real SDK code against a mocked Anthropic HTTP layer, no key needed:**
  - The agent's tool loop, server tools, fallback beta, adaptive thinking, cache control and cost
    math.
  - Structured output parsing, the Batches API flow and refusal handling.
- **Real worker run with no keys:**
  - Jobs that need Claude fail with "set ANTHROPIC_API_KEY in server/.env".
  - Email lookups fail with "set HUNTER_API_KEY …".
  - An approved email at 4 AM was scheduled for 9 AM, not sent.
- **Not run live, because these need your keys or accounts:**
  - real Claude calls (agent, writer, batches)
  - Hunter
  - Gmail sending and reply checks
  - Playwright on a real application form

## 4. What's needed now

1. **The profile.** Copy `server/jarvie/career/career_profile.example.toml` to
   `~/jarvie-data/career_profile.toml`, then fill it in:
   - graduation month, fields and companies to watch (watchlist)
   - under `[applicant]`: name and email, needed for signing emails and filling forms
   - your resumes under `[resumes.<name>]` (the example has swe, mle and qr), or a single
     `resume = ...` plus `resume_pdf` under `[applicant]`
   - add `quant` to `fields` if you want quant roles
2. **Each resume as text** at its `text` path, Markdown or `.txt`. The models read this
   version. Its `pdf` is what the apply assistant uploads; models don't read PDFs yet.
3. **Models.** By default the AI jobs run on your subscriptions, not the API, so they
   cost nothing per job (`[models]` in the profile; copy it from the example):
   - `research = "codex"`: scoring, searches and people finding on ChatGPT Plus (Codex CLI)
   - `documents = "claude-code:opus"`: cover letters and essays on Claude Pro (Claude Code)
   - `writer = "claude-code:sonnet"`: outreach emails
   - Sign in once: run `codex` and `claude` in a terminal. `"api"` uses the Anthropic API
     instead (needs `ANTHROPIC_API_KEY`). `"codex,claude-code:sonnet"` falls back to Claude
     when ChatGPT's limit is hit.
   - When a plan's limit is hit, the job goes back in the queue and AI jobs pause for 15
     minutes (`JARVIE_LIMIT_PAUSE_MIN`); free jobs keep running.
   - **Compare them on your postings:** `.venv/bin/python -m jarvie.career.compare --n 10`
     scores the same postings with each backend without changing your saved scores; read
     them side by side on the dashboard's Compare tab, then set `research` to the one you trust.
4. **Keys in `server/.env`** (add them yourself):
   - `ANTHROPIC_API_KEY`: only if a `[models]` entry is `"api"`
   - `HUNTER_API_KEY`: email lookup. The free tier is small (about 25 lookups a month).
5. **Optional, for the apply assistant:** `cd server && uv sync --extra apply &&
   .venv/bin/playwright install chromium`. This downloads about 150 MB of browser into
   `~/Library/Caches/ms-playwright`.
6. **Later, to send real email:**
   - Your Google Cloud OAuth client needs the Gmail API enabled, and the `gmail.send` and
     `gmail.readonly` scopes on its consent screen, with you as a test user.
   - Then run `.venv/bin/python -m jarvie.outreach.sender --auth` once. This creates
     `~/jarvie-data/gmail_token.json`, separate from the calendar token.
   - Then set `CAREER_SEND_ENABLED=1`.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Claude (worker only) |
| `HUNTER_API_KEY` | none | Email finder and verifier |
| `CAREER_SEND_ENABLED` | off | `1` sends from Gmail; otherwise a dry run to `~/jarvie-data/outbox_dry_run.jsonl` |
| `CAREER_DAILY_SEND_CAP` | 15 | Emails per day |
| `CAREER_JOB_BUDGET_USD` | 3.00 | Stop a search after spending this much |
| `JARVIE_DAILY_BUDGET_USD` | 10.00 | Worker stops claiming jobs after this much today |
| `JARVIE_RESEARCH_MODEL` / `JARVIE_WRITER_MODEL` | `claude-opus-5-5` / `claude-sonnet-5-5` | Model overrides |

## 5. Work in progress / not yet verified

- **Real Claude behaviour:**
  - The agent has been tested with fakes and mocked HTTP only.
  - Prompt quality (scoring calibration, how well it finds people, email tone) needs a few real
    runs and probably prompt tuning.
  - The cost estimates use list prices, which are an approximation.
- **The apply assistant's form reader** (`FIELDS_JS` in `career/apply_assist.py`) hasn't run in
  a real browser yet. Run the skipped test after installing Chromium, then try one real
  Greenhouse posting.
- **Gmail** sending and reply/bounce detection are unit-tested only.
- **Resume as PDF:** the models read the text version only.
- **No "summary" voice tool** for "what changed this week". Status history is stored, but no
  tool reads it across postings yet.

## 6. Known issues and edge cases

- **Merging into `main`:** `server/pyproject.toml` and `server/uv.lock` conflict with
  `semantic-scholar`, because both add `httpx` at the same line.
  - To fix: keep this branch's dependency list (it's a superset; the conflict is the `jinja2`
    line) and run `cd server && uv lock`.
  - `uv.lock` also conflicts with `wakeword-eval`. Same fix: `uv lock`.
  - No code files conflict, and I verified the merged result passes all tests.
- **The worker runs one thing at a time.** While a long search runs (minutes), scheduled emails
  wait until it finishes. That's fine at 15 a day, but a send can be a few minutes late.
- **Duplicate postings across sources:** the same job on Simplify and on Greenhouse has
  different URLs, so it's saved twice.
- **Resume auto-pick is keyword-based:** a "Quant Developer" posting tags both quant and
  software engineering; the title decides, and a tie goes to `default_resume`. Check the
  Resume line on a posting page before writing a letter, and pick one if it's wrong. The
  agent's suggestion during scoring never overrides your pick.
- **"trading" tags quant:** an "Energy Trading Operations" internship gets tagged quant.
- **Keyword tagging is rough:** "Product Management Intern – Data and AI" gets tagged as machine
  learning because of "AI". The agent's scoring is meant to correct this; the hard filter is
  only a first cut.
- **Lever and Ashby company names** are the board slug (`palantir`, `openai`), not the display
  name.
- **SimplifyJobs:** the `Summer2027-Internships` URL currently returns the same data as
  `Summer2026`, apparently a redirect. Its listings carry their own terms, which the term filter
  uses, so this is harmless. Check the repo name each season.
- **Graduation filtering works at the year level:** "class of 2027" passes for a May 2028
  graduation with a ±6-month window. It only drops a posting when no year mentioned is in range.
- **Too many tools for the voice model:** it now has 31 tools (36 once `semantic-scholar` is
  merged). gpt-4.1 handles that, but tool choice and latency may get worse. The README's planned
  two-stage tool retrieval would help.
- **Subscription backends:** each job runs `claude -p` or `codex exec` once, with web
  search only (no shell, no file edits, empty temp folder, nothing saved). They can't call
  Jarvie's tools mid-run, so they return the calls as JSON and Jarvie runs them through the
  same checks; a call that fails validation gets one repair round. Runs take ~20s+ each, and
  usage counts against the same limits as your own Claude Code / ChatGPT use. Job costs show
  $0 because nothing is billed per call.
- **The worker must be running for anything to happen.** Jarvie only queues jobs. The worker
  writes a heartbeat every 10 seconds; when it's missing, Jarvie says so when it queues
  something or reports status, and every dashboard page shows a banner with the start command.
- **What you say in a search request replaces the profile's pre-filled answers** (e.g.
  "biotech" replaces your profile's field list). Company lists hold names only;
  descriptions like "promising startups" go to notes. Company names match loosely
  ("Johnson & Johnson" = "Johnson and Johnson", "Eli Lilly and Company" = "Eli Lilly"), but
  naming a company only ranks it first: it's searched only if its board is on your watchlist.
- **Voice vs typed is a guess:** the model decides whether you typed or spoke an answer.
  Nothing depends on it; it's recorded for reference only.
- **A search that fails partway keeps what it saved:** for example, with no key, or when the
  budget runs out. Job-board postings are saved before scoring starts, so the job can show
  "failed" while new unscored postings appear. Re-running scores them.
- **Dry-run sends count as sent:** they get `dryrun-` ids. Turning on live sending later
  doesn't re-send them, so for real emails, draft again or add a reset.
- **Send times use your time zone,** not the recipient's.
- **Reply detection:** any message in the thread from someone else counts as a reply. An
  auto-reply ("out of office") counts as a reply too.
- **Hunter:** low free quota; work emails for small labs and high schools are often missing.
  Expect few Newton North matches; high-school alumni rarely appear on public pages.
- **Grounding checks catch common mistakes, not every one:** invented facts, placeholders,
  "your talk/paper" without a source, numbers not in your resume. Read every draft and letter;
  the dashboard flags what it can.
- **Forms:** some React-based forms ignore programmatic filling, and Greenhouse boards embedded
  on company sites run inside iframes. The assistant opens the board URL directly. Ashby's
  `/application` URL is a guess based on its common URL pattern.
- **Policy and etiquette:**
  - Keep the daily cap low and every email personal.
  - Don't scrape Handshake or LinkedIn; the code refuses both.
  - Contacts are personal data about third parties, stored only in your local database.
  - `do_not_contact` is permanent. Use it whenever someone asks you to stop.

## 7. Try it now (no keys needed)

Run these from `server/`. Use three terminals for the parts that need servers.

**Step 0: profile and resume (2 minutes)**
```bash
cp jarvie/career/career_profile.example.toml ~/jarvie-data/career_profile.toml
open -e ~/jarvie-data/career_profile.toml   # set graduation, fields, [applicant] name + email
# put each resume's text and PDF at its paths, e.g. ~/jarvie-data/resume_swe.md + .pdf
# add a few watchlist slugs, e.g. greenhouse = ["anthropic", "databricks"], ashby = ["openai"]
```

**Step 1: tests**
```bash
.venv/bin/python -m pytest -q          # expect: 208 passed, 1 skipped
```

**Step 2: pull real postings (free, about 30 seconds)**
```bash
.venv/bin/python -m jarvie.career.ingest --max 50 --term "Summer 2027"
# prints: fetched N, saved up to 50, and why the rest were dropped
```

**Step 3: dashboard.** Terminal 1: `.venv/bin/python -m jarvie.dashboard`, then open
http://localhost:7861.
- **Postings:** filter by field, open one, set its status to `applied` with a note, and mark a
  coffee chat.
- **Pipeline:** your counts by stage.
- **Searches:** start `discover_postings`. Your profile answers are pre-filled; type a term,
  answer `no` to search beyond the boards and `none` for AI scoring (a free search), and
  save. Clicking Start search with a required answer missing shows an error. Then submit.
- **Contacts:** add someone yourself, for example a friend's address you control.
- **Jobs:** the search you submitted is waiting for the worker.

**Step 4: voice and text.** Terminal 2: `uv run bot.py`. Open http://localhost:7860/client and
connect. Try:
- "How's my internship search going?" → `pipeline_summary`
- "Show me machine learning internships." → `list_postings`
- "I applied to the first one." → `set_application_status`
- "Find me research positions in biotech." → `start_brief`. Jarvie asks the questions; answer
  some by voice and type the rest in the "Type message…" box. Then refresh the dashboard's
  Searches tab: it's the same brief.
- Paste a Handshake posting's text into the text box with "add this posting" → `add_posting`
  queues a scoring job.

**Step 5: worker without keys.** Terminal 3: `.venv/bin/python -m jarvie.worker --once`. On the
Jobs page:
- the search from step 3 shows **failed**, with "set ANTHROPIC_API_KEY in server/.env". It
  fails at the scoring step, but the postings it pulled from the job boards are already saved:
  they appear on Postings, unscored.
- email lookups fail with "set HUNTER_API_KEY"

## 8. Tomorrow, with keys (first real run)

Add `ANTHROPIC_API_KEY` and `HUNTER_API_KEY` to `server/.env`, start the worker without
`--once`, and keep the dashboard open.

1. **Score one posting, about $0.05–0.30.** Paste a posting's text: "add this posting: …". On
   the Postings tab it should show scores with reasons.
2. **A small search, about $0.50–3.** Run a `discover_postings` brief with
   `max_results = 8`, AI scoring `all`, and search beyond the boards `yes`. Check the Jobs tab for cost, then read the reasons on a few postings.
3. **People.** Run a `find_contacts` brief: target = one company, roles = "university recruiter",
   schools = berkeley, max 3, draft after = no. Check that every contact has source links and
   that the emails show as verified.
4. **Drafts.** Say "draft coffee chat emails to them", then read the drafts on the Outreach tab
   and edit or reject them.
5. **Dry-run send.** Approve one. Between 9 and 5 on a weekday it "sends" to
   `~/jarvie-data/outbox_dry_run.jsonl` (labelled DRY RUN on the Outreach tab).
6. **Real send, to yourself first.** Do the Gmail setup (section 4.5), add yourself as a
   contact, draft, approve, set `CAREER_SEND_ENABLED=1`, and restart the worker.
7. **Cover letter.** Click "Write with Claude" on a posting, then check the flagged claims on
   the document page.
8. **Apply assistant**, after installing Chromium: run
   `.venv/bin/python -m jarvie.career.apply_assist <id>` for a Greenhouse posting. Review every
   field and submit yourself.
