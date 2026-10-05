import pytest

from jarvie import note

pytestmark = pytest.mark.usefixtures("temp_db")


def ids(items: list[dict]) -> list[int]:
    return [item["id"] for item in items]


def test_add_then_open_notes():
    new_id = note.add("fix wake word latency", category="bug")
    items = note.open_notes()
    assert ids(items) == [new_id]
    assert items[0]["text"] == "fix wake word latency"
    assert items[0]["category"] == "bug"
    assert items[0]["done_at"] is None


def test_add_defaults_category_to_feature():
    note.add("idea")
    assert note.open_notes()[0]["category"] == "feature"


def test_open_notes_newest_first_and_limit():
    first = note.add("first")
    second = note.add("second")
    third = note.add("third")
    assert ids(note.open_notes()) == [third, second, first]
    assert ids(note.open_notes(limit=2)) == [third, second]


def test_complete_note_moves_between_statuses():
    open_id = note.add("still open")
    done_id = note.add("finished")
    assert note.complete_note(done_id) is True

    assert ids(note.open_notes("open")) == [open_id]
    done = note.open_notes("done")
    assert ids(done) == [done_id]
    assert done[0]["done_at"] is not None
    assert set(ids(note.open_notes("all"))) == {open_id, done_id}


def test_complete_note_unknown_id():
    assert note.complete_note(9999) is False


def test_open_notes_rejects_unknown_status():
    with pytest.raises(ValueError):
        note.open_notes("bogus")


def test_edit_notes_updates_fields():
    new_id = note.add("old text", category="bug")
    assert note.edit_notes(new_id, text="new text", category="idea") is True
    item = note.open_notes()[0]
    assert item["text"] == "new text"
    assert item["category"] == "idea"


def test_edit_notes_unknown_id():
    assert note.edit_notes(9999, text="x") is False


def test_edit_notes_no_fields_is_noop():
    new_id = note.add("keep me")
    assert note.edit_notes(new_id) is False
    assert note.open_notes()[0]["text"] == "keep me"


def test_edit_notes_rejects_disallowed_field():
    new_id = note.add("protected")
    with pytest.raises(ValueError):
        note.edit_notes(new_id, done_at="2026-01-01")
