"""Sending approved emails: scheduling, sending, and noticing replies and bounces.

Only emails you approved on the dashboard are ever sent. Sending is a dry run (nothing
leaves your machine; would-be sends are written to ~/jarvie-data/outbox_dry_run.jsonl)
until CAREER_SEND_ENABLED=1 is set, and then goes out from your own Gmail.

Approved emails get send times spread over weekday business hours, at most
CAREER_DAILY_SEND_CAP per day (default 15), 20-60 minutes apart, so a batch doesn't
land all at once or trip Gmail's spam limits.

One-time Gmail sign-in (opens a browser):  .venv/bin/python -m jarvie.outreach.sender --auth
"""

import base64
import json
import os
import random
from datetime import datetime, time, timedelta
from email.mime.text import MIMEText
from typing import Protocol

from loguru import logger

from .. import config
from ..config import TZ
from . import contacts, messages

WINDOW_START, WINDOW_END = time(9, 0), time(17, 0)  # your local time zone (config.TZ)
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",  # to notice replies and bounces
]


class Mailer(Protocol):
    dry_run: bool

    def send(self, to: str, subject: str, body: str) -> tuple[str, str]:
        """Send one email; return (message_id, thread_id)."""
        ...

    def thread_state(self, thread_id: str) -> str | None:
        """'replied', 'bounced', or None if nothing new."""
        ...


class DryRunMailer:
    """Records what would be sent, sends nothing."""

    dry_run = True

    def __init__(self, path=None):
        self.path = path or (config.DATA_DIR / "outbox_dry_run.jsonl")

    def send(self, to, subject, body):
        n = random.randrange(10**8)
        with open(self.path, "a") as f:
            f.write(
                json.dumps(
                    {"to": to, "subject": subject, "body": body, "at": datetime.now(TZ).isoformat()}
                )
                + "\n"
            )
        return f"dryrun-{n}", f"dryrun-thread-{n}"

    def thread_state(self, thread_id):
        return None


class GmailMailer:
    """Sends from your Gmail via the Gmail API, with its own token (gmail_token.json),
    separate from the read-only calendar token, so calendar access is unaffected."""

    dry_run = False

    def __init__(self):
        self._service = None
        self._me = None

    def service(self):
        if self._service is None:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            token = config.DATA_DIR / "gmail_token.json"
            creds = None
            if token.exists():
                creds = Credentials.from_authorized_user_file(str(token), GMAIL_SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(config.GOOGLE_CREDENTIALS), GMAIL_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                token.write_text(creds.to_json())
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    def me(self) -> str:
        if self._me is None:
            self._me = self.service().users().getProfile(userId="me").execute()["emailAddress"]
        return self._me

    def send(self, to, subject, body):
        msg = MIMEText(body)
        msg["To"], msg["Subject"] = to, subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = self.service().users().messages().send(userId="me", body={"raw": raw}).execute()
        return sent["id"], sent["threadId"]

    def thread_state(self, thread_id):
        thread = (
            self.service()
            .users()
            .threads()
            .get(userId="me", id=thread_id, format="metadata", metadataHeaders=["From"])
            .execute()
        )
        return classify_thread(thread, self.me())


def classify_thread(thread: dict, me: str) -> str | None:
    """From a Gmail thread (metadata format): did anyone reply, or did it bounce?"""
    senders = [
        next(
            (h["value"] for h in m.get("payload", {}).get("headers", []) if h["name"] == "From"), ""
        )
        for m in thread.get("messages", [])[1:]
    ]
    if any("mailer-daemon" in s.lower() or "postmaster" in s.lower() for s in senders):
        return "bounced"
    if any(me.lower() not in s.lower() for s in senders if s):
        return "replied"
    return None


def default_mailer() -> Mailer:
    if os.getenv("CAREER_SEND_ENABLED") == "1":
        return GmailMailer()
    return DryRunMailer()


def daily_cap() -> int:
    return int(os.getenv("CAREER_DAILY_SEND_CAP", "15"))


def _next_slot(t: datetime) -> datetime:
    """t, moved forward into the next weekday sending window if it's outside one."""
    while True:
        if t.weekday() >= 5:  # Saturday/Sunday -> Monday morning
            t = datetime.combine(t.date() + timedelta(days=7 - t.weekday()), WINDOW_START, TZ)
        elif t.time() < WINDOW_START:
            t = datetime.combine(t.date(), WINDOW_START, TZ)
        elif t.time() >= WINDOW_END:
            t = datetime.combine(t.date() + timedelta(days=1), WINDOW_START, TZ)
        else:
            return t


def schedule_approved(now: datetime | None = None, rng: random.Random | None = None) -> list[int]:
    """Give every approved email a send time. Returns the ids scheduled."""
    now = now or datetime.now(TZ)
    rng = rng or random.Random()
    taken = [
        datetime.fromisoformat(m["scheduled_at"])
        for m in messages.queue()
        if m["status"] in ("scheduled", "sent") and m["scheduled_at"]
    ]
    per_day: dict = {}
    for t in taken:
        per_day[t.date()] = per_day.get(t.date(), 0) + 1
    cursor = max([now, *taken])
    scheduled = []
    for m in messages.queue("approved"):
        t = _next_slot(
            cursor + timedelta(minutes=rng.randint(20, 60)) if taken or scheduled else cursor
        )
        while per_day.get(t.date(), 0) >= daily_cap():
            t = _next_slot(datetime.combine(t.date() + timedelta(days=1), WINDOW_START, TZ))
        messages.set_status(m["id"], "scheduled", scheduled_at=t.isoformat())
        per_day[t.date()] = per_day.get(t.date(), 0) + 1
        cursor = t
        scheduled.append(m["id"])
    return scheduled


def send_due(mailer: Mailer, now: datetime | None = None) -> list[int]:
    """Send every scheduled email whose time has come. Returns the ids sent."""
    now = now or datetime.now(TZ)
    sent = []
    for m in messages.queue("scheduled"):
        if datetime.fromisoformat(m["scheduled_at"]) > now:
            continue
        if contacts.get(m["contact_id"])["do_not_contact"]:
            messages.set_status(m["id"], "stopped")
            continue
        try:
            message_id, thread_id = mailer.send(m["email"], m["subject"], m["body"])
        except Exception as e:  # Gmail errors vary; keep it scheduled and retry next tick
            logger.error("sending outreach {} failed: {}", m["id"], e)
            continue
        messages.set_status(
            m["id"],
            "sent",
            sent_at=datetime.now(TZ).isoformat(),
            gmail_message_id=message_id,
            gmail_thread_id=thread_id,
        )
        logger.info(
            "outreach {} sent{} to {}", m["id"], " (dry run)" if mailer.dry_run else "", m["email"]
        )
        sent.append(m["id"])
    return sent


def check_replies(mailer: Mailer) -> dict[int, str]:
    """Mark sent emails replied or bounced. Returns {outreach_id: new status}."""
    changed = {}
    for m in messages.queue("sent"):
        if not m["gmail_thread_id"] or m["gmail_thread_id"].startswith("dryrun-"):
            continue
        state = mailer.thread_state(m["gmail_thread_id"])
        if state:
            messages.set_status(m["id"], state)
            changed[m["id"]] = state
    return changed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Outreach sending utilities.")
    parser.add_argument("--auth", action="store_true", help="sign in to Gmail once")
    args = parser.parse_args()
    if args.auth:
        print("Signed in as", GmailMailer().me())


if __name__ == "__main__":
    main()
