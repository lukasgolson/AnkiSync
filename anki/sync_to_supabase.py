import sqlite3
import os
import json
import sys
import argparse
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ANKI_DB_PATH = os.getenv("ANKI_DB_PATH")

# Safety check
if not all([SUPABASE_URL, SUPABASE_KEY, ANKI_DB_PATH]):
    print("Error: Missing required environment variables in .env file.")
    sys.exit(1)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def safe_json(data_str):
    """Safely parse Anki's custom data strings into JSON for Supabase"""
    if not data_str or not data_str.strip():
        return None
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def sync_anki_to_supabase(dry_run=False):
    if dry_run:
        print("🟡 RUNNING IN DRY-RUN (DRAFT) MODE: No data will be written to Supabase.\n")
    else:
        print("🟢 RUNNING IN LIVE (PUBLISH) MODE: Data will be written to Supabase.\n")

    if not os.path.exists(ANKI_DB_PATH):
        print(f"Error: Could not find Anki database at {ANKI_DB_PATH}")
        sys.exit(1)

    try:
        conn = sqlite3.connect(ANKI_DB_PATH)
        cursor = conn.cursor()
    except sqlite3.OperationalError as e:
        print(f"Database Error: {e}")
        print("Make sure Anki is completely CLOSED before running this script!")
        sys.exit(1)

    # ==========================================
    # 1. SYNC DECKS (Full Sync)
    # ==========================================
    decks_payload = []

    try:
        # Modern Anki Schema (v15+)
        cursor.execute("SELECT id, name FROM decks")
        decks_data = cursor.fetchall()
        decks_payload = [{"id": row[0], "name": row[1]} for row in decks_data]
    except sqlite3.OperationalError:
        # Fallback for older Anki versions
        cursor.execute("SELECT decks FROM col")
        col_data = cursor.fetchone()
        if col_data and col_data[0]:
            decks_json = json.loads(col_data[0])
            decks_payload = [{"id": int(d_id), "name": d_info["name"]} for d_id, d_info in decks_json.items()]

    if decks_payload:
        if not dry_run:
            supabase.table('decks').upsert(decks_payload).execute()
            print(f"✅ Synced {len(decks_payload)} decks.")
        else:
            print(f"🟡 [DRY RUN] Would sync {len(decks_payload)} decks.")
    else:
        print("➖ No decks found.")

    # ==========================================
    # 2. SYNC NOTES (Delta Sync)
    # ==========================================
    notes_res = supabase.table('notes').select('mod').order('mod', desc=True).limit(1).execute()
    last_note_mod = notes_res.data[0]['mod'] if notes_res.data else 0

    cursor.execute("SELECT id, sfld, tags, mod FROM notes WHERE mod > ?", (last_note_mod,))
    modified_notes = cursor.fetchall()

    if modified_notes:
        notes_payload = [{"id": n[0], "sfld": n[1], "tags": n[2], "mod": n[3]} for n in modified_notes]
        if not dry_run:
            supabase.table('notes').upsert(notes_payload).execute()
            print(f"✅ Upserted {len(notes_payload)} modified notes.")
        else:
            print(f"🟡 [DRY RUN] Would upsert {len(notes_payload)} modified notes.")
    else:
        print("➖ No new/modified notes to sync.")

    # ==========================================
    # 3. SYNC CARDS (Delta Sync)
    # ==========================================
    cards_res = supabase.table('cards').select('mod').order('mod', desc=True).limit(1).execute()
    last_card_mod = cards_res.data[0]['mod'] if cards_res.data else 0

    cursor.execute("""
        SELECT id, nid, did, queue, type, ivl, factor, reps, lapses, mod, data 
        FROM cards WHERE mod > ?
    """, (last_card_mod,))
    modified_cards = cursor.fetchall()

    if modified_cards:
        cards_payload = [
            {
                "id":  c[0], "nid": c[1], "did": c[2], "queue": c[3], "type": c[4],
                "ivl": c[5], "factor": c[6], "reps": c[7], "lapses": c[8],
                "mod": c[9], "fsrs_data": safe_json(c[10])  # Safely parse FSRS data
            } for c in modified_cards
        ]
        if not dry_run:
            supabase.table('cards').upsert(cards_payload).execute()
            print(f"✅ Upserted {len(cards_payload)} modified cards.")
        else:
            print(f"🟡 [DRY RUN] Would upsert {len(cards_payload)} modified cards.")
    else:
        print("➖ No modified cards to sync.")

    # ==========================================
    # 4. SYNC REVLOG (Append-only Delta Sync)
    # ==========================================
    revlog_res = supabase.table('revlog').select('id').order('id', desc=True).limit(1).execute()
    last_revlog_id = revlog_res.data[0]['id'] if revlog_res.data else 0

    cursor.execute("""
        SELECT id, cid, ease, ivl, lastIvl, factor, time, type 
        FROM revlog WHERE id > ?
    """, (last_revlog_id,))
    new_reviews = cursor.fetchall()

    if new_reviews:
        revlog_payload = [
            {
                "id":      r[0], "cid": r[1], "ease": r[2], "ivl": r[3],
                "lastivl": r[4], "factor": r[5], "time": r[6], "type": r[7]
            } for r in new_reviews
        ]
        if not dry_run:
            supabase.table('revlog').insert(revlog_payload).execute()
            print(f"✅ Inserted {len(revlog_payload)} new reviews.")
        else:
            print(f"🟡 [DRY RUN] Would insert {len(revlog_payload)} new reviews.")
    else:
        print("➖ No new reviews to sync.")

    conn.close()
    print("\nSync process finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Anki database to Supabase.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the script in draft mode without writing to Supabase.")
    args = parser.parse_args()
    sync_anki_to_supabase(dry_run=args.dry_run)