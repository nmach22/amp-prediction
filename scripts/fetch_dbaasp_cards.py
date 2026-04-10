"""
DBAASP Full Data Fetcher (with MIC scores)
==========================================
Strategy:
  1. Load unique peptide IDs from a pre-downloaded stub CSV
     (data/raw/dbaasp_raw.csv, column ``id`` or ``peptide_id``).
  2. Fetch full peptide cards concurrently (MAX_WORKERS threads).
     Each card is appended to a JSONL checkpoint file immediately so
     an interrupted run can resume without re-fetching completed IDs.
  3. Flatten the activity records and persist final results to:
       - data/raw/dbaasp_full.json        (raw card objects, one per peptide)
       - data/raw/dbaasp_full.csv         (one row per activity entry)
       - data/raw/dbaasp_checkpoint.jsonl (incremental resume file)

Endpoint used:
  - Card : GET https://dbaasp.org/peptides/<peptide_id>

Tuning:
  - MAX_WORKERS : number of concurrent threads (default 8).
  - DELAY       : per-worker sleep between requests in seconds (default 0.2).

Prerequisites:
  Run ``fetch_dbaasp_sequences.py`` first to produce dbaasp_raw.csv.

Run:
  python scripts/fetch_dbaasp_cards.py
"""
import os
import sys
import time
import json
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from urllib3 import Retry

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import get_logger  # noqa: E402
log = get_logger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL    = "https://dbaasp.org/peptides"
DELAY       = 0.2   # seconds between requests per worker (be polite to the server)
MAX_WORKERS = 8     # concurrent threads; tune to avoid rate-limiting
PATH_DATA   = os.path.join(ROOT, "data", "raw")
OUTPUT_JSON = os.path.join(PATH_DATA, "dbaasp_full.json")
OUTPUT_CSV  = os.path.join(PATH_DATA, "dbaasp_full.csv")
CHECKPOINT  = os.path.join(PATH_DATA, "dbaasp_checkpoint.jsonl")  # one card per line
# ────────────────────────────────────────────────────────────────────────────

def fetch_peptide_card(peptide_id: int, session: requests.Session) -> dict:
    """Fetch the full peptide card (including activities / MIC) for one ID."""
    resp = session.get(os.path.join(BASE_URL, str(peptide_id)), timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_mic_rows(card: dict) -> list[dict]:
    """
    Pull every activity record out of a peptide card and return a flat list.

    The card typically contains an 'activities' (or 'antimicrobialActivities')
    list. Each entry looks like:
      {
        "targetSpecies": {"name": "E. coli"},
        "activityMeasure": {"name": "MIC"},
        "activity": 4.0,
        "unit": {"name": "µg/mL"},
        ...
      }
    We keep all activity types (MIC, HC50, IC50, etc.) so nothing is lost.
    """
    rows = []
    peptide_id   = card.get("id")
    peptide_name = card.get("name", "")
    sequence     = card.get("sequence", "")

    # DBAASP v1 card nests activities under different keys depending on version
    activities = (
        card.get("targetActivities")
        or []
    )

    if not activities:
        # Still store the peptide even with no activity data
        rows.append({
            "peptide_id":       peptide_id,
            "peptide_name":     peptide_name,
            "sequence":         sequence,
            "target_species":   None,
            "activity_measure": None,
            "activity_value":   None,
            "concentration":    None,
            "unit":             None,
            "medium":           None,
            "cfu":              None,
        })
        return rows

    for act in activities:
        target = act.get("targetSpecies") or {}
        measure = act.get("activityMeasureGroup") or {}
        unit    = act.get("unit") or {}
        rows.append({
            "peptide_id":           peptide_id,
            "peptide_name":         peptide_name,
            "sequence":             sequence,
            "target_species":       target.get("name") if isinstance(target, dict) else str(target),
            "activity_measure":     measure.get("name") if isinstance(measure, dict) else str(measure),
            "activity_value":       act.get("activity") or 0.0,
            "concentration":        act.get("concentration") or 0.0,
            "unit":                 unit.get("name") if isinstance(unit, dict) else str(unit),
            "medium":               act.get("medium"),
            "cfu":                  act.get("cfu"),
        })
    return rows


# ── HTTP session with automatic retry ────────────────────────────────────────
def _make_session(retries: int = 5, backoff: float = 1.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({
        "Accept":     "application/json",
        "User-Agent": "amp-prediction-research/1.0",
    })
    session.verify = False          # DBAASP uses a self-signed certificate
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session

def load_peptide_ids(path: Path) -> list[dict]:
    """Load unique peptide IDs from the raw CSV stub file."""
    log.info("=== Step 1: loading peptide IDs from %s ===", path)
    if not path.exists():
        log.error("Cannot read %s", path)
        exit(1)

    df = pd.read_csv(path)
    if "id" in df.columns:
        id_col = "id"
    elif "peptide_id" in df.columns:
        id_col = "peptide_id"
    else:
        log.error(
            "Could not find an 'id' or 'peptide_id' column in %s. Available: %s",
            path, list(df.columns),
        )
        exit(1)

    stubs = df[[id_col]].drop_duplicates().rename(columns={id_col: "id"}).to_dict(orient="records")
    log.info("Loaded %d unique peptide IDs from CSV.", len(stubs))
    return stubs


def _load_checkpoint() -> set[int]:
    """Return set of peptide IDs already saved in the checkpoint file."""
    done: set[int] = set()
    if not Path(CHECKPOINT).exists():
        return done
    with open(CHECKPOINT) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    log.info("Checkpoint: %d peptides already fetched – skipping.", len(done))
    return done


def _append_checkpoint(card: dict) -> None:
    """Append a single card to the JSONL checkpoint file (thread-safe via GIL)."""
    with open(CHECKPOINT, "a") as fh:
        fh.write(json.dumps(card) + "\n")


def _fetch_one(pid: int, session: requests.Session) -> dict:
    """Fetch one card, sleep afterwards, and return it."""
    card = fetch_peptide_card(pid, session)
    time.sleep(DELAY)
    return card


def fetch_all_cards(stubs: list[dict], session: requests.Session) -> tuple[list[dict], list[dict], list[int]]:
    """Fetch full peptide cards for every stub ID concurrently.

    Resumes from CHECKPOINT so an interrupted run doesn't restart from scratch.

    Returns:
        full_cards: raw card dicts from the API.
        all_rows:   flattened activity rows ready for a DataFrame.
        failed_ids: IDs that raised an exception during fetching.
    """
    log.info("=== Step 2: fetching peptide cards (with MIC) ===")

    done_ids   = _load_checkpoint()
    pending    = [s for s in stubs if s.get("id") not in done_ids]
    log.info("%d peptides to fetch (%d already checkpointed).", len(pending), len(done_ids))

    full_cards: list[dict] = []
    all_rows:   list[dict] = []
    failed_ids: list[int]  = []

    # Read already-fetched cards from checkpoint
    if done_ids:
        with open(CHECKPOINT) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        card = json.loads(line)
                        full_cards.append(card)
                        all_rows.extend(extract_mic_rows(card))
                    except json.JSONDecodeError:
                        pass

    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for stub in pending:
            pid = stub.get("id")
            if pid is None:
                continue
            futures[executor.submit(_fetch_one, pid, session)] = pid

        with tqdm(total=len(futures), unit="peptide") as pbar:
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    card = future.result()
                    _append_checkpoint(card)
                    full_cards.append(card)
                    all_rows.extend(extract_mic_rows(card))
                except Exception as exc:
                    failed_ids.append(pid)
                    log.warning("Peptide %d failed – %s", pid, exc)
                finally:
                    pbar.update(1)

    return full_cards, all_rows, failed_ids


def save_outputs(full_cards: list[dict], all_rows: list[dict], failed_ids: list[int]) -> None:
    """Persist raw cards to JSON and flattened activity rows to CSV."""
    log.info("=== Step 3: saving outputs ===")

    with open(OUTPUT_JSON, "w") as fh:
        json.dump(full_cards, fh, indent=2)
    log.info("Raw cards → %s  (%d peptides)", OUTPUT_JSON, len(full_cards))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Flat CSV  → %s  (%d activity rows)", OUTPUT_CSV, len(df))

    if failed_ids:
        log.warning("%d peptides failed: %s ...", len(failed_ids), failed_ids[:20])


def main():
    path_data = ROOT / "data" / "raw" / "dbaasp_raw.csv"

    stubs                          = load_peptide_ids(path_data)
    session                        = _make_session()
    full_cards, all_rows, failed   = fetch_all_cards(stubs, session)
    save_outputs(full_cards, all_rows, failed)
    log.info("Done!")


if __name__ == "__main__":
    main()