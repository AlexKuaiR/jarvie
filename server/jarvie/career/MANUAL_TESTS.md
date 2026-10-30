# Manual test checklist: career search, outreach, and running things at once

Tick each box as you go. Every test says what to do and what you should see. If you
see something else, note the job id (from the Jobs tab) and the time; the worker
log at `~/jarvie-data/logs/worker.log` usually says why.

**Setup (once per session)**
- Terminal 1, the bot: `cd ~/pipecat-quickstart/server && uv run bot.py`, then open
  http://localhost:7860. Start each spoken request with "Jarvie" ("Jarvis" deliberately doesn't wake it).
- Terminal 2, the dashboard: `jdash`, or
  `cd ~/pipecat-quickstart/server && .venv/bin/python -m jarvie.dashboard`, then open
  http://localhost:7861.
- Terminal 3, for watching the worker: `tail -f ~/jarvie-data/logs/worker.log`. You
  don't start the worker yourself: it starts when a job is queued and stops after
  60 idle seconds.

**What these tests use**
- AI jobs (searches, scoring, drafts) run on your Claude Pro / ChatGPT Plus plans, not
  paid per call.
- Each email lookup uses one Hunter credit (the free plan gives 25 a month).
- Nothing is ever submitted to an employer.
- Nothing is emailed unless you finish Part E, which sends only to you and one
  friend who has agreed to it.

---

## A. Finding internships by voice

- [ ] **A1. A quick search with no AI.** Say: "Jarvie, find me paid software engineering
  internships for Summer 2027." When it asks about searching beyond the job boards and
  AI scoring, answer "no" and "none".
  - Expect: it reads back your answers. When you say "go", it says it's queued *and
    gives a job id*. If it says "starting" without a job id, that's the old bug again.
  - Check: on the Jobs tab, `discover_postings` goes queued, then running, then done
    in about a minute. The worker log shows "worker started", and new postings
    appear on the Postings tab.
- [ ] **A2. A search with AI scoring.** Say: "Find me two machine learning research
  internships, search beyond the boards, and score the top three."
  - Expect: it asks only what you didn't already say, and the job takes a few minutes.
  - Check: on the Postings tab, three postings have a fit score and a reason.
- [ ] **A3. Corrections mid-brief.** Start a search. Say "Fields are biotech", then
  "actually make that machine learning and biotech". Then "skip Merck and Johnson and
  Johnson", then "include big companies like Eli Lilly".
  - Expect: on the dashboard Searches tab, fields shows both, and the excluded
    companies are exactly two names, "Merck" and "Johnson and Johnson". Eli Lilly is
    under include, and "big companies" is in notes.
- [ ] **A4. Finish a voice search by typing.** Start a search by voice, stop halfway,
  then finish it in the localhost text box: "the term is Summer 2027, no AI".
  - Expect: it continues the same brief instead of opening a second one. The
    Searches tab shows one brief, last answered by text.
- [ ] **A5. Finding a posting by spoken words.** Say "What's the status of the NVIDIA
  Ignite SWE one?", then "Mark the Datacor one as not interested."
  - Expect: it finds each one the first time, even if speech-to-text mishears "SWE".
    If two postings match, it asks which.
- [ ] **A6. Adding a posting from pasted text.** Paste a posting's full text into the
  text box: "add this posting: <text>".
  - Expect: a job id. A minute later, "did that save?" gives the posting's name.
  - Then try a JavaScript-heavy careers link, like a Workday or NVIDIA page. That job
    should *fail* with a suggestion to paste the text, not show "done".
- [ ] **A7. Status updates.** Say "I applied to the Klaviyo one", then later "I got
  an OA for it".
  - Expect: the Pipeline tab shows the status moving to applied, then oa, with your
    note.

## B. Finding internships from the terminal (no voice)

- [ ] **B1. Job boards only, no model:**
  `cd ~/pipecat-quickstart/server && .venv/bin/python -m jarvie.career.ingest --term "Summer 2027" --max 20`
  - Expect: it prints counts saved and dropped (with reasons), and the Postings tab
    has the new ones.
- [ ] **B2. Running it twice is safe.** Run the same command again.
  - Expect: no duplicate postings. Postings already saved (same link) are updated in
    place, keeping your status and notes.
- [ ] **B3. Start a search from the dashboard.** On the Searches tab, start a new
  search, fill in the form, and click Start search.
  - Expect: it appears on the Jobs tab and runs, the same as a voice search.
- [ ] **B4. Retry a failed job.** Find a failed job on the Jobs tab (A6 makes one)
  and click Retry.
  - Expect: it runs again, and its error is replaced by the new result or error.
- [ ] **B5. Compare the two AI models** (optional, a few minutes):
  `.venv/bin/python -m jarvie.career.compare --n 3`
  - Expect: a link to the Compare tab with Codex and Claude scores side by side.
    Your real scores don't change.

## C. Drafting emails to people you added yourself

Use people you know and who won't mind: yourself, a friend, a TA you've talked to.
Drafting never sends anything.

- [ ] **C1. Add a contact by voice:** "Jarvie, add my friend Sam Lee, sam@example.com,
  at Berkeley, to my contacts" (use a real friend's details).
  - Expect: the Contacts tab shows Sam with the email and the source "user".
- [ ] **C2. Add a contact on the dashboard.** Contacts tab, then Add, with your own
  second address or your Berkeley email as the contact.
- [ ] **C3. Draft by voice:** "Draft coffee chat emails to Sam and me, using my ML
  resume."
  - Expect: a job id. The draft job finishes in about a minute, and drafts appear on
    the Outreach tab with status "drafted".
  - Check: the email's background matches your mle resume.
- [ ] **C4. Match a resume to each person.** Give one contact the role "Software
  Engineer" and the other "ML Engineer". Then say "draft emails to both, match the
  resume to each person."
  - Expect: the software one draws on your swe resume, and the ML one on your mle
    resume.
- [ ] **C5. Find an email.** Add a contact without an email, at a real company, e.g.
  a recruiter you met. Say "find their email". This uses one Hunter credit.
  - Expect: found (verified, or "accept_all", which means unconfirmed) or not found.
    Not found should be said plainly, not guessed.
- [ ] **C6. Edit and approve.** Edit a draft's text on the Outreach tab and save it.
  Don't approve yet; that's Part E.
- [ ] **C7. Do not contact.** Mark one contact "do not contact", then ask to draft to
  them.
  - Expect: they're skipped, with the reason "do not contact".
- [ ] **C8. Read a summary:** "What did you write to Sam?"
  - Expect: the gist and points read aloud, not the whole email.
- [ ] **C9. Rewrite together:** "Make Sam's email shorter and mention we met at the
  career fair."
  - Expect: it saves "met at the career fair" as a fact about Sam, then gives a
    rewrite job id. A minute later the Outreach tab shows the new draft, a new
    summary, and "2 versions" (or more), with "Use this version" on older ones.
  - Edit the draft by hand first, then ask for a rewrite: your edit should survive.
- [ ] **C10. A rewrite can't invent things.** Ask "say I loved her podcast" for
  someone with no podcast stored.
  - Expect: the rewrite is not used, with the reason shown on the Outreach tab, and
    the draft unchanged.

## D. The apply assistant (fills the form, never submits)

One-time install (adds the optional browser packages):
`cd ~/pipecat-quickstart/server && uv sync --extra apply && .venv/bin/playwright install chromium`

These two saved postings are low stakes: low or no fit for you, and on forms the
assistant supports. Filling a form without submitting sends the company nothing.

| Posting id | Company / role | Form |
|---|---|---|
| 5 | Datacor, Data Science Intern, Summer 2027 (fit 0.12) | Greenhouse |
| 30 | Cyvl, Data Operations Intern, Summer 2027 (unscored) | Ashby |

- [ ] **D1. Greenhouse form:** `.venv/bin/python -m jarvie.career.apply_assist 5`
  - Expect: a visible Chromium window on the form. Name, email, phone and links are
    filled, and the resume is attached. It takes a screenshot, then the terminal
    says to review and press Enter.
  - Check: demographic questions, checkboxes and custom questions are still blank.
    There's no click on Submit.
  - **Close the window without submitting** (press Enter in the terminal).
- [ ] **D2. Ashby form:** the same with `apply_assist 30`.
- [ ] **D3. Which resume it uploads.** In the posting's dashboard page, set its resume
  to "qr" and run D1 again.
  - Expect: the upload is the qr PDF. Set it back afterwards.
- [ ] **D4. It refuses what it should.** Run it on a Workday or Handshake posting, if
  you have one saved.
  - Expect: a clear refusal and no browser.

## E. Sending emails for real (to you and one willing friend only)

Do this on a weekday between 9 and 5 (your time). Emails sent outside that window wait
until the next window opens.

- [ ] **E1. Dry run first.** Approve your own draft from C2/C3 on the Outreach tab.
  - Expect: it becomes "scheduled" with a time, and when that time comes, "sent".
    Check `~/jarvie-data/outbox_dry_run.jsonl`: the email is there, and nothing
    arrived in your inbox.
  - The worker stays running while an email is waiting; you don't have to keep the
    bot open.
- [ ] **E2. Turn on real sending.**
  - Sign in to Gmail once: `.venv/bin/python -m jarvie.outreach.sender --auth`. A
    browser opens; it should print "Signed in as <you>".
  - Add `CAREER_SEND_ENABLED=1` to `server/.env`. Wait until the worker has stopped
    (the log says "worker stopped"); the next one reads the new setting.
- [ ] **E3. Send to yourself.** Draft and approve an email to your own second address.
  - Expect: it arrives from your Gmail at the scheduled time. The Outreach tab says
    sent.
- [ ] **E4. A reply is noticed.** Send one to the friend who agreed and ask them to
  reply.
  - Expect: within about 10 minutes of their reply (while the worker runs), the
    email shows "replied". Replies are only checked while the worker is running, so
    queue any job to wake it if needed.
- [ ] **E5. Spacing.** Approve two at once.
  - Expect: the second one is scheduled 20–60 minutes after the first.
- [ ] **E6. Turn sending back off** afterwards if you want: remove
  `CAREER_SEND_ENABLED=1`.

## F. Doing other things while a career job runs

Start a long job first: an AI search like A2 (several minutes). Watch the worker log
in Terminal 3. Then, while it says "running":

- [ ] **F1. The bot stays responsive.** Say "What's on my calendar tomorrow?", "Add a
  deadline: problem set 5 due Friday at 11:59 PM", and "Add a note: buy printer paper".
  - Expect: each answers in normal time; none waits for the search.
  - Check: the deadline and note show up when you ask for them.
- [ ] **F2. Career questions mid-job.** Say "How's my search going?" and "What are my
  top postings?"
  - Expect: the status is "running" (it doesn't claim done), and postings saved so
    far are listed.
- [ ] **F3. Email lookups queue behind the search.** Say "Find the email for <a
  contact>."
  - Expect: you get a job id right away and the status says "queued". The worker runs
    one job at a time, so the lookup starts after the search finishes, then takes a
    few seconds. Jarvie should say it's waiting, not that it failed.
- [ ] **F4. A second search while one runs.** Start another small search (A1 style).
  - Expect: it queues and runs after the first. The worker log shows no second
    "worker started" (one worker only).
- [ ] **F5. Terminal and dashboard at the same time.** While the job runs, run
  B1 (`ingest`) in another terminal and change a posting's status on the dashboard.
  - Expect: no "database is locked" errors. Everything ends up saved.
- [ ] **F6. Closing the bot doesn't stop the job.** Stop the bot (Ctrl+C) mid-search.
  - Expect: the job still finishes (see the Jobs tab). Restart the bot and ask "is my
    search done?". It finds the finished job.
- [ ] **F7. Disconnect and reconnect.** Reload the localhost page mid-job, then ask for
  the job's status.
  - Expect: it can still report on the job; jobs live in the database, not the
    conversation.
  - Also try muting and unmuting the mic. If speech-to-text stops after unmuting,
    note whether the bot terminal printed "Media stream error while reading the
    audio; clearing track." Reloading the page is the workaround for now.

---

**When you find a problem**, send me the test number, what you said or ran, the job id,
and the time. The worker log and the database have the rest.
