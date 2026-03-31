"""
scripts/fetch_dbaasp_sequences.py
==========================
Downloads all peptide sequences from the DBAASP REST API (v4) and saves a
flat CSV to data/raw/ ready for make_splits.py.
API reference : https://dbaasp.org/api?page=rest
OpenAPI spec  : https://dbaasp.org/v3/api-docs
What gets downloaded
--------------------
Endpoint  GET /peptides
Params    limit=<page_size>  offset=<N>  complexity.value=monomer
Response  {"totalCount": N, "data": [...]}
Each record in data[] has:
  id, dbaaspId, name, sequence, sequenceLength, complexity, synthesisType
All DBAASP entries are experimentally confirmed antimicrobial peptides, so
every downloaded sequence receives  activity = 1  (AMP = positive class).
For a binary classifier you still need a *negative* (non-AMP) set.
Common choices: randomly sampled UniProt/Swiss-Prot sequences not present
in any AMP database, or purpose-built negative sets from APD3/CAMP.
Output CSV  (data/raw/<out>)
----------------------------
  dbaasp_id   DBAASP identifier  (e.g. "DBAASPS_8")
  sequence    amino-acid sequence (uppercase; may contain non-standard AA)
  activity    binary label: always 1 for this dataset
  name        peptide name (may be empty)
  synthesis   synthesis type (Ribosomal / Synthetic / …)
Usage
-----
  # recommended: start here to confirm field names
  python scripts/fetch_dbaasp_sequences.py --dry-run
  # full download (~24k peptides, ~5 min at --delay 0.2)
  python scripts/fetch_dbaasp_sequences.py
  # custom output / page size / delay
  python scripts/fetch_dbaasp_sequences.py --out dbaasp_raw.csv --page-size 500 --delay 0.1
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# DBAASP uses a self-signed TLS certificate.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── project root & shared logger ─────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import get_logger  # noqa: E402
log = get_logger(__name__)

# ── API constants (from  https://dbaasp.org/v3/api-docs) ─────────────────────
BASE_URL    = "https://dbaasp.org"
PEPTIDE_EP  = "/peptides"          # GET  ?limit=N&offset=N[&complexity.value=monomer]
DEFAULT_PAGE_SIZE = 500            # server seems happy up to 1000

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
    return session

# ── single-page fetch ─────────────────────────────────────────────────────────
def _fetch_page(
    session: requests.Session,
    limit: int,
    offset: int,
    extra_params: dict | None = None,
    timeout: int = 60,
) -> dict:
    params = {"limit": limit, "offset": offset}
    if extra_params:
        params.update(extra_params)

    resp = session.get(BASE_URL + PEPTIDE_EP, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# ── core download function ────────────────────────────────────────────────────
def download(
    out_file: str = "dbaasp_raw.csv",
    page_size: int = DEFAULT_PAGE_SIZE,
    delay: float = 0.2,
    monomers_only: bool = False,
    dry_run: bool = False,
) -> Path:
    """Download DBAASP peptides and save to data/raw/<out_file>.
    Args:
        out_file:       Output filename inside data/raw/.
        page_size:      Records per API request.
        delay:          Seconds to sleep between requests.
        monomers_only:  If True, only monomer-complexity records are fetched
                        (skips multimers/multi_peptides entirely).
        dry_run:        Fetch 2 records, print raw JSON, and exit.
    Returns:
        Path to the saved CSV.
    """
    out_path = ROOT / "data" / "raw" / out_file
    session  = _make_session()
    extra = {"complexity.value": "monomer"} if monomers_only else {}

    # ── dry-run ───────────────────────────────────────────────────────────────
    if dry_run:
        log.info("DRY-RUN — fetching 2 records from DBAASP to show raw JSON …")
        data = _fetch_page(session, limit=2, offset=0, extra_params=extra)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        log.info("totalCount reported by server: %s", data.get("totalCount"))
        return out_path

    # ── discover total ────────────────────────────────────────────────────────
    log.info("Probing DBAASP for total record count …")
    first = _fetch_page(session, limit=1, offset=0, extra_params=extra)
    total = int(first.get("totalCount") or 0)
    if total == 0:
        log.error("Server returned totalCount=0. Check connectivity.")
        sys.exit(1)
    total_pages = -(-total // page_size)   # ceiling division
    log.info(
        "Total records: %d  |  page size: %d  |  pages: %d",
        total, page_size, total_pages,
    )

    # ── paginate ──────────────────────────────────────────────────────────────
    records: list[dict] = []
    for page_idx in range(total_pages):
        offset = page_idx * page_size
        try:
            page_data = _fetch_page(session, limit=page_size, offset=offset,
                                    extra_params=extra)
        except requests.HTTPError as exc:
            log.warning("HTTP error at offset %d: %s — retrying after 5 s", offset, exc)
            time.sleep(5)
            page_data = _fetch_page(session, limit=page_size, offset=offset,
                                    extra_params=extra)

        records.extend(page_data.get("data", []))

        progress_page = page_idx + 1
        if progress_page % 5 == 0 or progress_page == total_pages:
            log.info(
                "  page %4d / %d  |  rows collected: %d",
                progress_page, total_pages, len(records),
            )
        if page_idx < total_pages - 1:
            time.sleep(delay)

    # ── save ──────────────────────────────────────────────────────────────────
    if not records:
        log.error("No records collected. Aborting.")
        sys.exit(1)
    df = pd.DataFrame(records).drop_duplicates(subset="sequence", keep="first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved %d unique sequences → %s", len(df), out_path)
    log.info("  activity=1 (AMP)  : %d (100 %%)", len(df))
    log.info(
        "NOTE: All DBAASP entries are positive AMPs. "
        "Add a negative (non-AMP) set before training a binary classifier."
    )
    log.info("Next step → python scripts/make_splits.py --input %s", out_file)
    return out_path

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download DBAASP peptide sequences via REST API v4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out", default="dbaasp_raw.csv",
        help="Output filename written to data/raw/.",
    )
    p.add_argument(
        "--page-size", type=int, default=DEFAULT_PAGE_SIZE,
        help="Records fetched per API request.",
    )
    p.add_argument(
        "--delay", type=float, default=0.2,
        help="Sleep (s) between requests — be polite to the server.",
    )
    p.add_argument(
        "--monomers-only", action="store_true",
        help="Only fetch monomer-complexity records (skip multimers).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch 2 records, print raw JSON, exit. Confirm fields before a full run.",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    download(
        out_file=args.out,
        page_size=args.page_size,
        delay=args.delay,
        monomers_only=args.monomers_only,
        dry_run=args.dry_run,
    )
