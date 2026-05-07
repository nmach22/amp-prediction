"""
Gram Stain Classifier for AMP Target Organisms
------------------------------------------------
Classifies bacteria (and other organisms) from DBAASP into:
  - gram_positive
  - gram_negative
  - non_bacteria   (viruses, fungi, parasites, etc.)
  - unknown        (ambiguous or unrecognised names)

Usage:
    python classify_gram_stain.py

Requires:
    pip install anthropic pandas tqdm
    export ANTHROPIC_API_KEY="sk-..."
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV   = "../../data/raw/amp_mic_activities.csv"
OUTPUT_CSV  = "../../data/processed/amp_mic_activities_gram_classified.csv"
TARGET_COL  = "target_activity_name"
BATCH_SIZE  = 80          # organisms per API call  (~80 is safe for context)
MODEL       = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_DELAY = 5           # seconds between retries
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a microbiology expert. 
Your only job is to classify organism names into exactly one of these four categories:
  - gram_positive   : gram-positive bacteria
  - gram_negative   : gram-negative bacteria  
  - non_bacteria    : viruses, fungi, yeasts, parasites, archaea, mammalian cells, or any non-bacterial organism
  - unknown         : genuinely ambiguous, unrecognised, or insufficient information

Rules:
- Use the genus name as the primary signal (e.g. Staphylococcus → gram_positive, Escherichia → gram_negative).
- Strain suffixes (ATCC 6538, DH5alpha, etc.) do not change the gram classification.
- If a name contains both a clear genus AND a conflicting hint, trust the genus.
- Respond ONLY with a valid JSON object — no markdown, no explanation, no extra text.
  Format: {"organism_name": "category", ...}
"""

def classify_batch(client: anthropic.Anthropic, names: list[str]) -> dict[str, str]:
    """Send one batch to Claude and return {name: category} dict."""
    numbered = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names))
    user_msg = (
        f"Classify each organism below. "
        f"Return a JSON object where every key is the exact organism name "
        f"and the value is one of: gram_positive, gram_negative, non_bacteria, unknown.\n\n"
        f"{numbered}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)

            # Validate all names are present
            missing = [n for n in names if n not in result]
            if missing:
                print(f"  ⚠ Missing {len(missing)} names in response — marking as 'unknown'")
                for m in missing:
                    result[m] = "unknown"

            return result

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ✗ Parse error on attempt {attempt}: {e}")
        except anthropic.RateLimitError:
            print(f"  ⏳ Rate limit — waiting {RETRY_DELAY * attempt}s …")
            time.sleep(RETRY_DELAY * attempt)
        except Exception as e:
            print(f"  ✗ Unexpected error on attempt {attempt}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # All retries exhausted — fall back to unknown
    print(f"  ✗ All {MAX_RETRIES} attempts failed. Marking batch as 'unknown'.")
    return {n: "unknown" for n in names}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from: {INPUT_CSV}")
    amp_df = pd.read_csv(INPUT_CSV)
    unique_names = amp_df[TARGET_COL].dropna().unique().tolist()
    print(f"  → {len(unique_names)} unique organism names to classify")

    # ── Batch classify ─────────────────────────────────────────────────────
    results: dict[str, str] = {}
    batches = [unique_names[i:i+BATCH_SIZE] for i in range(0, len(unique_names), BATCH_SIZE)]

    print(f"\nClassifying in {len(batches)} batches of ≤{BATCH_SIZE} …\n")
    for batch in tqdm(batches, desc="Classifying"):
        batch_result = classify_batch(client, batch)
        results.update(batch_result)
        time.sleep(0.3)   # gentle throttle

    # ── Merge back ────────────────────────────────────────────────────────
    gram_series = amp_df[TARGET_COL].map(results)
    amp_df["gram_classification"] = gram_series

    # Summary
    print("\n── Classification summary ──────────────────────────────")
    print(amp_df["gram_classification"].value_counts(dropna=False).to_string())
    print("────────────────────────────────────────────────────────\n")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    amp_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved enriched DataFrame to: {OUTPUT_CSV}")

    # Also save the lookup table for reuse
    lookup_path = OUTPUT_CSV.replace(".csv", "_lookup.csv")
    pd.DataFrame(results.items(), columns=[TARGET_COL, "gram_classification"]) \
      .to_csv(lookup_path, index=False)
    print(f"✓ Saved lookup table to:        {lookup_path}")


if __name__ == "__main__":
    main()