import sqlite3
import os
import json
from supabase import create_client, Client

# --- Configuration ---
# Adjust this path to point to your actual Anki collection.anki2 file
ANKI_DB_PATH = os.path.expanduser("~/AppData/Roaming/Anki2/User 1/collection.anki2") 
SUPABASE_URL = "https://your-project-ref.supabase.co"
SUPABASE_KEY = "your-anon-or-service-role-key"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def sync_anki_to_supabase():
    # Connect to local Anki SQLite DB (Make sure Anki is closed!)
    conn = sqlite3.connect(ANKI_DB_PATH)
    cursor = conn.cursor()

    print("Starting comprehensive Anki sync...")

    # ==========================================
    # 1. SYNC DECKS (Full Sync)
    # ==========================================
    # Decks are stored as a JSON blob in the 'col' table. It's tiny, so we just overwrite it.
    cursor.execute("SELECT decks FROM col")
    col_data = cursor.fetchone()
    if col_data:
        decks_json = json.loads(col_data[0])
        decks_payload = [
            {"id": int(deck_id), "name": deck_info["name"]} 
            for deck_id, deck_info in decks_json.items()
        ]
        supabase.table('decks').upsert(decks_payload).execute()
        print(f"Synced {len(decks_payload)} decks.")

    # ==========================================
    # 2. SYNC NOTES (Delta Sync)
    # ==========================================
    notes_res = supabase.table('notes').select('mod').order('mod', desc=True).limit(1).execute()
    last_note_mod = notes_res.data[0]['mod'] if notes_res.data else 0

    cursor.execute("""
        SELECT id, sfld, tags, mod 
        FROM notes 
        WHERE mod > ?
    """, (last_note_mod,))
    modified_notes = cursor.fetchall()

    if modified_notes:
        notes_payload = [
            {"id": n[0], "sfld": n[1], "tags": n[2], "mod": n[3]} 
            for n in modified_notes
        ]
        # Upsert: Update if exists, insert if new
        supabase.table('notes').upsert(notes_payload).execute()
        print(f"Upserted {len(notes_payload)} modified notes.")
    else:
        print("No new/modified notes to sync.")

    # ==========================================
    # 3. SYNC CARDS (Delta Sync)
    # ==========================================
    cards_res = supabase.table('cards').select('mod').order('mod', desc=True).limit(1).execute()
    last_card_mod = cards_res.data[0]['mod'] if cards_res.data else 0

    # We've added 'nid' (Note ID) to this query
    cursor.execute("""
        SELECT id, nid, did, queue, type, ivl, factor, reps, lapses, mod, data 
        FROM cards 
        WHERE mod > ?
    """, (last_card_mod,))
    modified_cards = cursor.fetchall()

    if modified_cards:
        cards_payload = [
            {
                "id": c[0], "nid": c[1], "did": c[2], "queue": c[3], "type": c[4], 
                "ivl": c[5], "factor": c[6], "reps": c[7], "lapses": c[8], 
                "mod": c[9], "fsrs_data": c[10]
            } for c in modified_cards
        ]
        supabase.table('cards').upsert(cards_payload).execute()
        print(f"Upserted {len(cards_payload)} modified cards.")
    else:
        print("No modified cards to sync.")

    # ==========================================
    # 4. SYNC REVLOG (Append-only Delta Sync)
    # ==========================================
    revlog_res = supabase.table('revlog').select('id').order('id', desc=True).limit(1).execute()
    last_revlog_id = revlog_res.data[0]['id'] if revlog_res.data else 0

    cursor.execute("""
        SELECT id, cid, ease, ivl, lastIvl, factor, time, type 
        FROM revlog 
        WHERE id > ?
    """, (last_revlog_id,))
    new_reviews = cursor.fetchall()

    if new_reviews:
        revlog_payload = [
            {
                "id": r[0], "cid": r[1], "ease": r[2], "ivl": r[3], 
                "lastIvl": r[4], "factor": r[5], "time": r[6], "type": r[7]
            } for r in new_reviews
        ]
        # Insert: Revlogs are just a history log, no need to upsert
        supabase.table('revlog').insert(revlog_payload).execute()
        print(f"Inserted {len(revlog_payload)} new reviews.")
    else:
        print("No new reviews to sync.")

    conn.close()
    print("Sync complete!")

if __name__ == "__main__":
    sync_anki_to_supabase()