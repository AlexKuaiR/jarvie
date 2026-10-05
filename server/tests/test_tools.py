import asyncio
from datetime import datetime, timedelta

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from jarvie import calendar, deadlines, note, tools
from jarvie.config import TZ
from jarvie.db import connect

pytestmark = pytest.mark.usefixtures("temp_db")


class FakeParams:
    """Stands in for Pipecat's FunctionCallParams: just records what the tool reports back."""

    def __init__(self):
        self.result = None

    async def result_callback(self, result):
        self.result = result


def call(tool, **kwargs) -> dict:
    """Run an async tool to completion and return what it reported.

    The tools are async; asyncio.run drives one without needing a pytest plugin.
    """
    params = FakeParams()
    asyncio.run(tool(params, **kwargs))
    return params.result


def in_days(n: int) -> str:
    return (datetime.now(TZ).date() + timedelta(days=n)).isoformat()


# --- what the LLM sees -------------------------------------------------------


def test_tool_schemas_required_fields():
    # Pipecat builds each tool's schema from its signature: no default = required.
    schemas = LLMContext(tools=tools.ALL_TOOLS).tools.standard_tools
    required = {s.name: s.required for s in schemas}
    assert required == {
        "get_events": [],
        "get_deadlines": [],
        "add_deadlines": ["name", "date"],
        "complete_deadline": ["item_id"],
        "mute_deadline": ["item_id"],
        "update_deadline": ["item_id"],
        "get_note": [],
        "add_note": ["text"],
        "update_note": ["item_id"],
        "complete_note": ["item_id"],
    }


# --- briefing ----------------------------------------------------------------


def test_briefing_none_when_nothing_due():
    assert tools.briefing() is None


def test_briefing_mentions_upcoming_item():
    deadlines.add(name="Midterm 1", date=in_days(2))
    brief = tools.briefing()
    assert "Midterm 1" in brief
    assert "mute_deadline" in brief


def test_briefing_mentions_overdue_item():
    deadlines.add(name="Chem homework", date=in_days(-3))
    brief = tools.briefing()
    assert "Chem homework" in brief
    assert "complete_deadline" in brief


def test_briefing_asks_to_mention_upcoming_alongside_overdue():
    # Regression: with both kinds present, only the overdue part told the model to
    # speak, so the upcoming item was never brought up.
    deadlines.add(name="Chem homework", date=in_days(-3))
    deadlines.add(name="Data 88 project", date=in_days(5))
    brief = tools.briefing()
    assert "Chem homework" in brief
    assert "Data 88 project" in brief
    assert "Mention" in brief


# --- deadline tools ----------------------------------------------------------


def test_get_deadlines_empty():
    assert call(tools.get_deadlines) == {"found": False, "items": [], "overdue": []}


def test_add_deadlines_then_get():
    result = call(tools.add_deadlines, name="Essay", date=in_days(3), kind="assignment")
    assert result["added"] is True

    got = call(tools.get_deadlines)
    assert got["found"] is True
    assert [(i["id"], i["name"], i["kind"]) for i in got["items"]] == [
        (result["id"], "Essay", "assignment")
    ]


def test_get_deadlines_separates_overdue_from_upcoming():
    late = deadlines.add(name="Chem homework", date=in_days(-2))
    soon = deadlines.add(name="Essay", date=in_days(3))

    got = call(tools.get_deadlines)
    assert got["found"] is True
    assert [i["id"] for i in got["items"]] == [soon]
    assert [(i["id"], i["days_overdue"]) for i in got["overdue"]] == [(late, 2)]


def test_get_deadlines_found_with_only_overdue():
    deadlines.add(name="Chem homework", date=in_days(-1))
    got = call(tools.get_deadlines)
    assert got["found"] is True
    assert got["items"] == []


def test_complete_deadline_clears_overdue():
    item_id = deadlines.add(name="Chem homework", date=in_days(-2))
    assert call(tools.complete_deadline, item_id=item_id) == {"completed": True}
    assert call(tools.get_deadlines)["overdue"] == []


def test_add_deadlines_saves_prep_hours():
    result = call(tools.add_deadlines, name="Final", date=in_days(5), prep_hours_needed=6.0)
    # upcoming() doesn't return prep hours, so check the row directly.
    with connect() as conn:
        row = conn.execute(
            "SELECT prep_hours_needed FROM deadlines WHERE id = ?", (result["id"],)
        ).fetchone()
    assert row["prep_hours_needed"] == 6.0


def test_mute_deadline_hides_it():
    item_id = deadlines.add(name="Lab report", date=in_days(4))
    assert call(tools.mute_deadline, item_id=item_id) == {"muted": True}
    assert call(tools.get_deadlines)["found"] is False


def test_complete_deadline_hides_it():
    item_id = deadlines.add(name="Problem set", date=in_days(2))
    assert call(tools.complete_deadline, item_id=item_id) == {"completed": True}
    assert call(tools.get_deadlines)["found"] is False


def test_unknown_id_reports_false():
    assert call(tools.mute_deadline, item_id=9999) == {"muted": False}
    assert call(tools.complete_deadline, item_id=9999) == {"completed": False}
    assert call(tools.update_deadline, item_id=9999, date=in_days(1)) == {"updated": False}


def test_update_deadline_without_name_keeps_name():
    item_id = deadlines.add(name="Midterm 1", date="2099-01-10")

    assert call(tools.update_deadline, item_id=item_id, date="2099-01-12") == {"updated": True}

    items = deadlines.upcoming(days=365 * 100)
    assert [(i["name"], i["date"]) for i in items] == [("Midterm 1", "2099-01-12")]


# --- get_events (calendar faked out; no Google calls) ------------------------


def test_get_events_asks_for_the_right_day(monkeypatch):
    seen = {}

    def fake_between(start, end):
        seen["start"], seen["end"] = start, end
        return [{"title": "Chem lecture"}]

    monkeypatch.setattr(calendar, "between", fake_between)

    result = call(tools.get_events, day_offset=1)

    tomorrow = datetime.now(TZ).date() + timedelta(days=1)
    assert result == {
        "date": tomorrow.isoformat(),
        "found": True,
        "events": [{"title": "Chem lecture"}],
    }
    assert seen["start"].date() == tomorrow
    assert seen["start"].hour == 0
    assert seen["end"] - seen["start"] == timedelta(days=1)


def test_get_events_reports_calendar_failure(monkeypatch):
    def broken(start, end):
        raise RuntimeError("token expired")

    monkeypatch.setattr(calendar, "between", broken)

    assert call(tools.get_events) == {"found": False, "error": "Calendar unavailable."}


def test_get_note_lists_open_notes():
    new_id = note.add("log wake word misses")
    result = call(tools.get_note)
    assert result["found"] is True
    assert [item["id"] for item in result["items"]] == [new_id]


def test_get_note_bad_status_returns_error_instead_of_raising():
    result = call(tools.get_note, status="pending")
    assert "error" in result
    assert "pending" in result["error"]
