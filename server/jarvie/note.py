from datetime import datetime

from .config import TZ
from .db import connect


def add(text: str, category: str = "feature") -> int:
    now = datetime.now(TZ).isoformat()
    if not text:
        raise ValueError("no text inputted")
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO dev_notes
                (text, category, created_at)
                VALUES (?, ?, ?)""",
            (text, category, now),
        )
        return cur.lastrowid


def open_notes(status: str = "open", limit: int = 20) -> list[dict]:
    sql = "SELECT * FROM dev_notes"
    if status == "open":
        sql += " WHERE done_at is NULL ORDER BY created_at DESC LIMIT ?"
    elif status == "done":
        sql += " WHERE done_at is NOT NULL ORDER BY created_at DESC LIMIT ?"
    elif status == "all":
        sql += " ORDER BY created_at DESC LIMIT ?"
    else:
        raise ValueError(f"cannot open status {status}")
    with connect() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "category": r["category"],
            "created_at": r["created_at"],
            "done_at": r["done_at"],
        }
        for r in rows
    ]


def edit_notes(item_id: int, **fields: str | None) -> bool:
    allowed = {"text", "category"}
    sets, values = [], []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"cannot update {key}")
        sets.append(f"{key} = ?")
        values.append(value)

    if not sets:
        return False

    values.append(item_id)
    sql = f"UPDATE dev_notes SET {', '.join(sets)} WHERE id = ?"
    with connect() as conn:
        cur = conn.execute(sql, values)
        return cur.rowcount > 0


def complete_note(item_id: int) -> bool:
    sql = "UPDATE dev_notes SET done_at = ? WHERE id = ?"
    with connect() as conn:
        return conn.execute(sql, (datetime.now(TZ).isoformat(), item_id)).rowcount > 0
