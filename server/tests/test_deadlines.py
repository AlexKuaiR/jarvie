from datetime import datetime, timedelta

import pytest

from jarvis import deadlines
from jarvis.config import TZ

pytestmark = pytest.mark.usefixtures("temp_db")


def in_days(n: int) -> str:
    return (datetime.now(TZ).date() + timedelta(days=n)).isoformat()


def ids(items: list[dict]) -> list[int]:
    return [item["id"] for item in items]


def test_add_then_upcoming():
    new_id = deadlines.add(name="Midterm 1", date=in_days(3), course="Chem")
    items = deadlines.upcoming()
    assert ids(items) == [new_id]
    assert items[0]["name"] == "Midterm 1"
    assert items[0]["days_away"] == 3


def test_upcoming_excludes_past_and_beyond_horizon():
    deadlines.add(name="Past", date=in_days(-1))
    deadlines.add(name="Far", date=in_days(10))
    inside = deadlines.add(name="Soon", date=in_days(7))
    assert ids(deadlines.upcoming(days=7)) == [inside]


def test_complete_hides_item():
    new_id = deadlines.add(name="Essay", date=in_days(2))
    assert deadlines.complete(new_id) is True
    assert deadlines.upcoming() == []
    assert deadlines.complete(9999) is False


def test_mute_hides_unless_due_within_a_day():
    later = deadlines.add(name="Later", date=in_days(5))
    tomorrow = deadlines.add(name="Tomorrow", date=in_days(1))
    deadlines.mute(later, days=3)
    deadlines.mute(tomorrow, days=3)

    assert ids(deadlines.upcoming()) == [tomorrow]
    assert set(ids(deadlines.upcoming(include_muted=True))) == {later, tomorrow}


def test_update():
    new_id = deadlines.add(name="Quiz", date=in_days(2))
    assert deadlines.update(new_id, date=in_days(4)) is True
    assert deadlines.upcoming()[0]["days_away"] == 4

    with pytest.raises(ValueError):
        deadlines.update(new_id, completed_at="now")
    assert deadlines.update(new_id) is False
