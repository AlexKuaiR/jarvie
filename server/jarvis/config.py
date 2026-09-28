from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")

DATA_DIR = Path.home() / "jarvis-data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "jarvis.db"
GOOGLE_CREDENTIALS = DATA_DIR / "credentials.json"
GOOGLE_TOKEN = DATA_DIR / "token.json"