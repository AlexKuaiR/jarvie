from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from .config import GOOGLE_CREDENTIALS, GOOGLE_TOKEN

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

_service_cache = None


def _service():
    global _service_cache
    if _service_cache is not None:
        return _service_cache
    creds = None
    if GOOGLE_TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(GOOGLE_CREDENTIALS), SCOPES)
            creds = flow.run_local_server(port=0)
        GOOGLE_TOKEN.write_text(creds.to_json())
    _service_cache = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _service_cache


def between(start: datetime, end: datetime) -> list[dict]:
    events = (
        _service()
        .events()
        .list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        )
        .execute()
        .get("items", [])
    )

    out = []
    for e in events:
        start_val = e["start"].get("dateTime") or e["start"].get("date")
        out.append(
            {
                "title": e.get("summary", "(no title)"),
                "start": start_val,
                "end": e["end"].get("dateTime") or e["end"].get("date"),
                "location": e.get("location"),
                "all_day": "dateTime" not in e["start"],
            }
        )
    return out


def upcoming(limit: int = 10) -> list[dict]:
    events = (
        _service()
        .events()
        .list(
            calendarId="primary",
            timeMin=datetime.now(timezone.utc).isoformat(),
            maxResults=limit,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
        .get("items", [])
    )

    out = []
    for e in events:
        start = e["start"].get("dateTime") or e["start"].get("date")
        out.append(
            {
                "title": e.get("summary", "(no title)"),
                "start": start,
                "end": e["end"].get("dateTime") or e["end"].get("date"),
                "location": e.get("location"),
                "description": e.get("description"),
                "all_day": "dateTime" not in e["start"],
            }
        )
    return out


if __name__ == "__main__":
    for ev in upcoming():
        print(ev)
