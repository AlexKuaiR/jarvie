from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")

DATA_DIR = Path.home() / "jarvie-data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "jarvie.db"
GOOGLE_CREDENTIALS = DATA_DIR / "credentials.json"
GOOGLE_TOKEN = DATA_DIR / "token.json"
