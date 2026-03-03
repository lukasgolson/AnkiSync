import os
import sys
import json
import argparse
import re
import random
from datetime import datetime, timedelta
import pytz

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

# Machine Learning Imports
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use Service Key to bypass RLS

if not all([SUPABASE_URL, SUPABASE_KEY]):
    print("Error: Missing required environment variables in .env file.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_local_today():
    return datetime.now(pytz.timezone('America/Vancouver')).date()


def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, ' ', raw_html).strip()


def clean_for_embeddings(text, tags):
    raw = f"{text} {tags}".replace('_', ' ')
    return re.sub(r'[^\w\s]', ' ', str(raw).lower())


def get_structural_features(text):
    text_str = str(text)
    word_count = len(text_str.split())
    char_count = len(text_str)
    has_cloze = 1 if 'c1::' in text_str or 'c2::' in text_str else 0
    return word_count, char_count, has_cloze


# ==========================================
# 2. DATA FETCHING (From Supabase)
# ==========================================
def fetch_all(table_name, select_query='*'):
    all_data = []
    page_size = 1000
    offset = 0
    while True:
        response = supabase.table(table_name).select(select_query).range(offset, offset + page_size - 1).execute()
        records = response.data
        all_data.extend(records)
        if len(records) < page_size:
            break
        offset += page_size
    return pd.DataFrame(all_data)


def load_data():
    print("Fetching raw data from Supabase...")
    decks = fetch_all('decks')
    notes = fetch_all('notes', 'id, sfld, tags')
    cards = fetch_all('cards')
    revlog = fetch_all('revlog')

    if cards.empty or revlog.empty:
        return None, None, None, None

    print("Cleaning and merging DataFrames...")
    vancouver_tz = pytz.timezone('America/Vancouver')

    revlog['review_datetime'] = pd.to_datetime(revlog['id'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
        vancouver_tz)
    revlog['review_date'] = revlog['review_datetime'].dt.date
    revlog['hour'] = revlog['review_datetime'].dt.hour
    revlog['ease_label'] = revlog['ease'].map({1: 'Again', 2: 'Hard', 3: 'Good', 4: 'Easy'})

    def parse_fsrs(fsrs_val):
        if not fsrs_val:
            return {"s": None, "d": None, "r": None}
        if isinstance(fsrs_val, str):
            try:
                return json.loads(fsrs_val)
            except:
                return {"s": None, "d": None, "r": None}
        return fsrs_val

    fsrs_df = cards['fsrs_data'].apply(parse_fsrs).apply(pd.Series)
    cards = pd.concat([cards, fsrs_df], axis=1)
    cards['s'] = pd.to_numeric(cards['s'], errors='coerce')
    cards['d'] = pd.to_numeric(cards['d'], errors='coerce')

    cards['creation_date'] = pd.to_datetime(cards['id'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
        vancouver_tz).dt.date
    cards = cards.merge(decks.rename(columns={'id': 'did', 'name': 'deck_name'}), on='did', how='left')
    cards = cards.merge(notes.rename(columns={'id': 'nid', 'sfld': 'card_front', 'tags': 'tags'}), on='nid', how='left')
    cards['clean_text'] = cards['card_front'].apply(clean_html)
    cards['deck_name'] = cards['deck_name'].apply(lambda x: x.replace('\x1f', ' ➔ ') if isinstance(x, str) else x)

    def get_state(row):
        if row['type'] == 0:
            return 'Unseen'
        elif row['type'] in [1, 3]:
            return 'Learning'
        elif row['type'] == 2:
            if row['ivl'] < 7:
                return 'Seen'
            elif row['ivl'] < 21:
                return 'Intermediate'
            else:
                return 'Known'
        return 'Other'

    cards['knowledge_state'] = cards.apply(get_state, axis=1)
    last_reviews = revlog.groupby('cid')['review_datetime'].max().reset_index().rename(
        columns={'review_datetime': 'last_review_datetime'})
    cards = cards.merge(last_reviews, left_on='id', right_on='cid', how='left')

    # Add calculated due date
    def calc_due(row):
        if pd.notnull(row['last_review_datetime']) and pd.notnull(row['ivl']) and row['type'] == 2:
            return (row['last_review_datetime'] + timedelta(days=row['ivl'])).date()
        return None

    cards['due_date'] = cards.apply(calc_due, axis=1)

    return decks, notes, cards, revlog


# ==========================================
# 3. RPG & TAMAGOTCHI ENGINE LOGIC
# ==========================================
def calculate_rpg_state(cards_df, revlog_df, db_stats):
    mature_count = len(cards_df[cards_df['knowledge_state'] == 'Known'])
    avg_s = cards_df['s'].mean() if 's' in cards_df.columns else 0

    total_xp = int((mature_count * 20) + (avg_s * 10))
    gross_gold = int(total_xp * 0.1)
    interest = int(gross_gold * 0.05) if avg_s > 21 else 0

    today_demotions = len(revlog_df[
                              (revlog_df['review_date'] == get_local_today()) &
                              (revlog_df['ease'] == 1) &
                              (revlog_df['type'] == 1)
                              ]) if 'type' in revlog_df.columns else 0

    daily_tax = today_demotions * 10
    bonus_gold = db_stats.get('bonus_gold', 0)
    spent_gold = db_stats.get('spent_gold', 0)

    raw_gold = (gross_gold + interest + bonus_gold) - (spent_gold + daily_tax)
    current_gold = max(0, raw_gold)

    user_level = int((total_xp / 150) ** 0.5) + 1
    fortitude_lvl = int(avg_s / 5)
    retention = (revlog_df['ease'] > 1).mean() * 100 if not revlog_df.empty else 0
    precision_lvl = int(retention / 10)
    avg_time = revlog_df['time'].mean() / 1000 if not revlog_df.empty else 0
    celerity_lvl = int(max(0, 100 - (avg_time * 5)) / 10)
    seen_cards = len(cards_df[cards_df['knowledge_state'] != 'Unseen'])
    cart_lvl = int((seen_cards / len(cards_df)) * 10) if len(cards_df) > 0 else 0

    return {
        "level": user_level,
        "xp":    total_xp,
        "gold":  current_gold,
        "radar": {
            "fortitude":   fortitude_lvl,
            "precision":   precision_lvl,
            "celerity":    celerity_lvl,
            "cartography": cart_lvl
        }
    }


# Pass db_stats in so we can read the potions bought from the HTML frontend!
def calculate_tamagotchi_state(cards_df, revlog_df):
    today = get_local_today()
    known_cards = len(cards_df[cards_df['knowledge_state'] == 'Known'])

    # ... (Evolution logic stays the same) ...
    if known_cards < 100:
        stage, sprite = "Egg", "🥚"
    elif known_cards < 500:
        stage, sprite = "Hatchling", "🐥"
    elif known_cards < 1500:
        stage, sprite = "Scholar-Beast", "🦉"
    else:
        stage, sprite = "Brain Dragon", "🐉"

    # ... (Raw Anki math stays the same) ...
    daily_count = 0
    streak = 0
    retention = 0

    if not revlog_df.empty:
        today_revs = revlog_df[revlog_df['review_date'] == today]
        daily_count = len(today_revs)
        if daily_count > 0:
            retention = (today_revs['ease'] > 1).mean()

        daily_totals = revlog_df.groupby('review_date').size().reset_index().sort_values('review_date', ascending=False)
        check_date = today
        if check_date not in daily_totals['review_date'].values:
            check_date -= timedelta(days=1)

        for d in daily_totals['review_date']:
            if d == check_date:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break

    # Calculate BASE raw stats
    base_hunger = int((daily_count / 50) * 100)
    base_health = int(retention * 100) if daily_count > 0 else 0
    base_happiness = streak * 10




    return {
        "metrics": {
            "hunger":      base_hunger,
            "health":      base_health,
            "happiness":   base_happiness,
            "streak":      streak,
            "known_cards": known_cards
        }
    }


# ==========================================
# 4. NEURAL ORACLE (ML PIPELINE)
# ==========================================
def run_ml_pipeline(cards_df, revlog_df):
    print("Running Neural Oracle (Embeddings & Gradient Boosting)...")
    ml_df = cards_df.copy()
    ml_df['nlp_ready'] = ml_df.apply(lambda x: clean_for_embeddings(x['clean_text'], x['tags']), axis=1)
    ml_df[['word_count', 'char_count', 'has_cloze']] = ml_df['card_front'].apply(
        lambda x: pd.Series(get_structural_features(x)))

    if not revlog_df.empty:
        time_stats = revlog_df.groupby('cid')['time'].mean().reset_index()
        time_stats['avg_time_sec'] = time_stats['time'] / 1000.0
        ml_df = ml_df.merge(time_stats[['cid', 'avg_time_sec']], left_on='id', right_on='cid', how='left')

    predictions_payload = {"unseen_threats": [], "cursed_due": []}

    if len(ml_df[ml_df['knowledge_state'] != 'Unseen']) > 50:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # --- Horizon Regressors ---
        train_reg = ml_df[
            (ml_df['knowledge_state'] != 'Unseen') & (ml_df['d'].notna()) & (ml_df['avg_time_sec'].notna())].copy()
        target_reg = ml_df[ml_df['knowledge_state'] == 'Unseen'].copy()

        if not target_reg.empty and not train_reg.empty:
            X_train_emb = embedder.encode(train_reg['nlp_ready'].tolist(), show_progress_bar=False)
            X_train_struct = train_reg[['word_count', 'char_count', 'has_cloze']].values
            X_train_combined = np.hstack((X_train_emb, X_train_struct))

            X_target_emb = embedder.encode(target_reg['nlp_ready'].tolist(), show_progress_bar=False)
            X_target_struct = target_reg[['word_count', 'char_count', 'has_cloze']].values
            X_target_combined = np.hstack((X_target_emb, X_target_struct))

            gbr_diff = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
            gbr_diff.fit(X_train_combined, train_reg['d'])
            target_reg['Predicted_Difficulty'] = gbr_diff.predict(X_target_combined)

            threats = target_reg.sort_values(by='Predicted_Difficulty', ascending=False).head(20)
            predictions_payload["unseen_threats"] = threats[
                ['clean_text', 'deck_name', 'Predicted_Difficulty']].to_dict('records')

        # --- Cursed Classifier ---
        train_clf = ml_df[ml_df['reps'] > 3].copy()
        train_clf['failure_rate'] = train_clf['lapses'] / train_clf['reps']
        train_clf['is_cursed'] = (train_clf['failure_rate'] > 0.15).astype(int)

        future_3_days = get_local_today() + timedelta(days=3)
        # Fix date comparison logic here
        due_target = ml_df[
            (ml_df['due_date'].notna()) & (pd.to_datetime(ml_df['due_date']).dt.date <= future_3_days) & (
                        ml_df['knowledge_state'] != 'Unseen')].copy()

        if not due_target.empty and train_clf['is_cursed'].sum() > 10:
            X_clf_emb = embedder.encode(train_clf['nlp_ready'].tolist(), show_progress_bar=False)
            X_clf_struct = train_clf[['word_count', 'char_count', 'has_cloze']].values
            X_train_clf = np.hstack((X_clf_emb, X_clf_struct))

            gbc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced',
                                                 random_state=42)
            gbc.fit(X_train_clf, train_clf['is_cursed'])

            X_due_emb = embedder.encode(due_target['nlp_ready'].tolist(), show_progress_bar=False)
            X_due_struct = due_target[['word_count', 'char_count', 'has_cloze']].values
            X_target_clf = np.hstack((X_due_emb, X_due_struct))

            due_target['Cursed_Probability'] = gbc.predict_proba(X_target_clf)[:, 1] * 100
            lapses = due_target[due_target['Cursed_Probability'] > 50].sort_values('Cursed_Probability',
                                                                                   ascending=False)
            predictions_payload["cursed_due"] = lapses[['clean_text', 'Cursed_Probability']].head(20).to_dict('records')

    return predictions_payload


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def generate_state(dry_run=False):
    if dry_run:
        print("🟡 RUNNING IN DRY-RUN MODE: No JSON state will be written to Supabase.\n")
    else:
        print("🟢 RUNNING IN LIVE MODE: Pushing calculated state to Supabase.\n")

    decks, notes, cards, revlog = load_data()
    if cards is None:
        print("Error: No data found in Supabase. Run sync_to_supabase.py first.")
        sys.exit(1)

    user_res = supabase.table('user_stats').select('*').eq('id', 'lukas').execute()
    db_stats = user_res.data[0] if user_res.data else {"spent_gold": 0, "bonus_gold": 0,
                                                       "inventory":  {"display_case": []}, "claimed_bounties": {}}

    print("Calculating RPG metrics...")
    char_stats = calculate_rpg_state(cards, revlog, db_stats)

    print("Calculating Tamagotchi Life Signs...")
    pet_stats = calculate_tamagotchi_state(cards, revlog)

    fragile_cards = cards[cards['s'] < 3]
    boss_data = {"active": False}
    if not fragile_cards.empty and 'deck_name' in fragile_cards.columns:
        bosses = fragile_cards.groupby('deck_name').agg(avg_s=('s', 'mean'), count=('id', 'count'),
                                                        total_d=('d', 'sum'))
        bosses = bosses[bosses['count'] > 10].sort_values('avg_s')
        if not bosses.empty:
            target_deck = bosses.index[0]
            boss_data = {
                "active":   True,
                "name":     f"{str(target_deck).split(' ➔ ')[-1].upper()} Guardian",
                "hp":       int(bosses.loc[target_deck, 'total_d'] * 10),
                "strategy": f"Search 'deck:{target_deck} prop:s<3' in Anki."
            }

    random.seed(get_local_today().toordinal())
    shop_data = [
        {"id": "w_daily", "name": "🗡️ The Peptide Scalpel", "type": "Main Hand", "cost": 50},
        {"id": "a_daily", "name": "🛡️ Leather Carapace", "type": "Armor", "cost": 100},
        {"id": "r_daily", "name": "🧭 Navigational Astrolabe", "type": "Relic", "cost": 75}
    ]

    ml_predictions = run_ml_pipeline(cards, revlog)

    master_state = {
        "last_updated": datetime.now(pytz.timezone('America/Vancouver')).isoformat(),
        "character":    char_stats,
        "pet":          pet_stats,
        "dungeon":      boss_data,
        "shop":         shop_data,
        "predictions":  ml_predictions
    }

    if not dry_run:
        supabase.table('rpg_state').upsert({"id": "lukas", "state_data": master_state}).execute()
        print(f"✅ Successfully crunched data and pushed Master State to Supabase.")
    else:
        print("\n🟡 [DRY RUN] Would push the following JSON payload:")
        print(json.dumps(master_state, indent=2, default=str)[:1500] + "\n... [TRUNCATED] ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crunch Anki Data and Generate RPG State.")
    parser.add_argument("--dry-run", action="store_true", help="Run ML and math without writing to DB.")
    args = parser.parse_args()
    generate_state(dry_run=args.dry_run)