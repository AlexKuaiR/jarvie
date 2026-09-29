from datetime import datetime

from jarvis import calendar
from jarvis.config import TZ


class FakeService:
    """Mimics the Google client's chained calls: service.events().list(...).execute()."""

    def __init__(self, items):
        self.items = items
        self.list_kwargs = None

    def events(self):
        return self

    def list(self, **kwargs):
        self.list_kwargs = kwargs
        return self

    def execute(self):
        return {"items": self.items}


def test_between_parses_timed_and_all_day_events(monkeypatch):
    service = FakeService(
        [
            {
                "summary": "Chem lecture",
                "start": {"dateTime": "2099-01-10T09:00:00-08:00"},
                "end": {"dateTime": "2099-01-10T10:00:00-08:00"},
                "location": "Room 101",
            },
            {
                # No title, all-day event: Google sends "date" instead of "dateTime".
                "start": {"date": "2099-01-10"},
                "end": {"date": "2099-01-11"},
            },
        ]
    )
    monkeypatch.setattr(calendar, "_service", lambda: service)

    start = datetime(2099, 1, 10, tzinfo=TZ)
    end = datetime(2099, 1, 11, tzinfo=TZ)
    events = calendar.between(start, end)

    assert events == [
        {
            "title": "Chem lecture",
            "start": "2099-01-10T09:00:00-08:00",
            "end": "2099-01-10T10:00:00-08:00",
            "location": "Room 101",
            "all_day": False,
        },
        {
            "title": "(no title)",
            "start": "2099-01-10",
            "end": "2099-01-11",
            "location": None,
            "all_day": True,
        },
    ]
    assert service.list_kwargs["timeMin"] == start.isoformat()
    assert service.list_kwargs["timeMax"] == end.isoformat()
    assert service.list_kwargs["singleEvents"] is True


def test_between_no_events(monkeypatch):
    monkeypatch.setattr(calendar, "_service", lambda: FakeService([]))
    start = datetime(2099, 1, 10, tzinfo=TZ)
    assert calendar.between(start, start) == []
