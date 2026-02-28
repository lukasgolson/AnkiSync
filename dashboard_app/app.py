import json
import re
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client
import pytz

from datetime import datetime

def get_local_today():
    """Always returns the current date in Vancouver, regardless of server location."""
    return datetime.now(pytz.timezone('America/Vancouver')).date()

GLOBAL_STOP_WORDS = set([
    'this', 'that', 'with', 'from', 'what', 'which', 'where', 'when', 'how', 'have',
    'will', 'about', 'these', 'those', 'their', 'there', 'they', 'because', 'other',
    'some', 'such', 'into', 'over', 'after', 'more', 'most', 'only', 'also', 'even',
    'very', 'just', 'like', 'well', 'many', 'much', 'each', 'every', 'both', 'between',
    'through', 'under', 'below', 'above', 'before', 'since', 'during', 'without',
    'within', 'among', 'along', 'against', 'until', 'despite', 'towards', 'moreau',
    'payette', 'bauce', 'allen', 'rademacher', 'results', 'discussion', 'background'
])

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Lukas Comprehensive Exam Dashboard", layout="wide", page_icon="🧠")
st.title("🧠 Lukas FSRS Comp Exam Prep Dashboard")


# ==========================================
# DATABASE CONNECTION & DATA FETCHING
# ==========================================
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_connection()


def clean_html(raw_html):
    """Strips HTML tags from Anki card text so the NLP engine only reads real words."""
    if not isinstance(raw_html, str):
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, ' ', raw_html).strip()


def fetch_all(table_name, select_query='*'):
    """Paginates through Supabase to bypass the 1,000 row API limit."""
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

@st.cache_data(ttl=3600)
def load_data():
    decks = fetch_all('decks')
    notes = fetch_all('notes', 'id, sfld, tags')
    cards = fetch_all('cards')
    revlog = fetch_all('revlog')

    if not revlog.empty:
        # 1. Localize epoch to UTC, then convert to Vancouver time
        vancouver_tz = pytz.timezone('America/Vancouver')
        revlog['review_datetime'] = pd.to_datetime(revlog['id'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
            vancouver_tz)

        revlog['review_date'] = revlog['review_datetime'].dt.date
        revlog['hour'] = revlog['review_datetime'].dt.hour
        ease_map = {1: 'Again', 2: 'Hard', 3: 'Good', 4: 'Easy'}
        revlog['ease_label'] = revlog['ease'].map(ease_map)

    def parse_fsrs(fsrs_val):
        if not fsrs_val:
            return {"s": None, "d": None, "r": None}
        if isinstance(fsrs_val, str):
            try:
                fsrs_val = json.loads(fsrs_val)
            except:
                return {"s": None, "d": None, "r": None}
        return fsrs_val

    def sanitize_deck_name(name):
        if not isinstance(name, str):
            return name
        # ASCII 31 is the '' character you're seeing
        clean_name = name.replace('\x1f', ' ➔ ')
        return clean_name

    if not cards.empty:
        fsrs_df = cards['fsrs_data'].apply(parse_fsrs).apply(pd.Series)
        cards = pd.concat([cards, fsrs_df], axis=1)
        cards['s'] = pd.to_numeric(cards['s'], errors='coerce')
        cards['d'] = pd.to_numeric(cards['d'], errors='coerce')

        vancouver_tz = pytz.timezone('America/Vancouver')
        cards['creation_date'] = pd.to_datetime(cards['id'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
            vancouver_tz).dt.date

        cards = cards.merge(decks.rename(columns={'id': 'did', 'name': 'deck_name'}), on='did', how='left')
        cards = cards.merge(notes.rename(columns={'id': 'nid', 'sfld': 'card_front', 'tags': 'tags'}), on='nid',
                            how='left')

        # Clean HTML for the NLP map
        cards['clean_text'] = cards['card_front'].apply(clean_html)
        cards['deck_name'] = cards['deck_name'].apply(sanitize_deck_name)

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

        if not revlog.empty:
            last_reviews = revlog.groupby('cid')['review_datetime'].max().reset_index()
            last_reviews = last_reviews.rename(columns={'review_datetime': 'last_review_datetime'})
            cards = cards.merge(last_reviews, left_on='id', right_on='cid', how='left')

            # 1. Predict Forgetting Date (When memory drops < 90%)
            cards['forgetting_date'] = cards['last_review_datetime'] + pd.to_timedelta(cards['s'], unit='D')
            cards['forgetting_date'] = cards['forgetting_date'].dt.date

            # 2. Calculate Actual Due Date based on Anki Interval
            def calc_due(row):
                if pd.notnull(row['last_review_datetime']) and pd.notnull(row['ivl']) and row['type'] == 2:
                    return (row['last_review_datetime'] + timedelta(days=row['ivl'])).date()
                return None

            cards['due_date'] = cards.apply(calc_due, axis=1)

    return decks, notes, cards, revlog


def get_user_stats():
    res = supabase.table('user_stats').select('*').eq('id', 'lukas').execute()
    return res.data[0] if res.data else None

def update_user_stats(updates):
    supabase.table('user_stats').update(updates).eq('id', 'lukas').execute()


def calculate_rpg_state(cards_df, revlog_df, db_stats):
    """Calculates the current state of the player based on deck performance and database ledger."""
    mature_count = len(cards_df[cards_df['knowledge_state'] == 'Known'])
    avg_s = cards_df['s'].mean() if 's' in cards_df.columns else 0

    total_xp = int((mature_count * 20) + (avg_s * 10))
    gross_gold = int(total_xp * 0.1)

    interest = int(gross_gold * 0.05) if avg_s > 21 else 0

    # --- FIXED TAX LOGIC: Only penalize actual demotions (Lapses) ---
    if 'type' in revlog_df.columns:
        # type == 1 means the card was in the "Review" phase.
        # ease == 1 means you hit "Again".
        today_demotions = len(revlog_df[
                                  (revlog_df['review_date'] == get_local_today()) &
                                  (revlog_df['ease'] == 1) &
                                  (revlog_df['type'] == 1)
                                  ])
    else:
        # Fallback just in case
        today_demotions = 0

    # Tax is 10 gold per actual memory lapse
    daily_tax = today_demotions * 10

    bonus_gold = db_stats.get('bonus_gold', 0)
    spent_gold = db_stats.get('spent_gold', 0)

    # Calculate exact net worth before applying the zero-floor
    raw_gold = (gross_gold + interest + bonus_gold) - (spent_gold + daily_tax)
    current_gold = max(0, raw_gold)
    tax_debt = abs(raw_gold) if raw_gold < 0 else 0

    return total_xp, current_gold, gross_gold, interest, daily_tax, avg_s, bonus_gold, tax_debt


with st.spinner("Loading & crunching Anki data..."):
    decks_df, notes_df, cards_df, revlog_df = load_data()

if cards_df.empty or revlog_df.empty:
    st.warning("No data found!")
    st.stop()

# ==========================================
# SIDEBAR FILTERS & EXAM COUNTDOWN
# ==========================================
st.sidebar.header("🎯 Target")
exam_date = date(get_local_today().year, 4, 14)
if get_local_today() > exam_date:
    exam_date = date(get_local_today().year + 1, 4, 14)

days_left = (exam_date - get_local_today()).days
st.sidebar.metric("Days Until Exam (Apr 14)", f"{days_left} Days")
st.sidebar.divider()

st.sidebar.header("Filters")
all_decks = ["All Decks"] + list(cards_df['deck_name'].dropna().unique())
selected_deck = st.sidebar.selectbox("Select a Deck", all_decks)

if selected_deck != "All Decks":
    filtered_cards = cards_df[cards_df['deck_name'] == selected_deck]
    filtered_revlog = revlog_df[revlog_df['cid'].isin(filtered_cards['id'])]
else:
    filtered_cards = cards_df
    filtered_revlog = revlog_df

state_colors = {
    'Unseen':       '#9ca3af', 'Learning': '#facc15', 'Seen': '#fb923c',
    'Intermediate': '#60a5fa', 'Known': '#22c55e'
}
ease_colors = {'Again': '#ef4444', 'Hard': '#f59e0b', 'Good': '#22c55e', 'Easy': '#3b82f6'}

# --- SIDEBAR PACE CALCULATOR ---
st.sidebar.divider()
st.sidebar.header("🏃‍♂️ Required Pace")

unseen_cards = len(filtered_cards[filtered_cards['knowledge_state'] == 'Unseen'])
if days_left > 0:
    daily_pace = unseen_cards / days_left
    st.sidebar.metric("New Cards Per Day", f"{daily_pace:.1f}",
                      help="Cards I must start daily to finish the deck.")
    if daily_pace > 30:
        st.sidebar.warning("⚠️ High Pace Required! Consider triaging 'Unseen' cards.")
    else:
        st.sidebar.success("✅ Pace is manageable.")
elif days_left == 0:
    st.sidebar.success("🎉 IT IS EXAM DAY! Good luck!")
else:
    st.sidebar.info("Exam has passed. Hope you crushed it!")


if st.sidebar.button("🔄 Clear Cache & Re-sync"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# TOP LEVEL METRICS
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flashcards", len(filtered_cards))

if not filtered_revlog.empty:
    total_reviews = len(filtered_revlog)
    passed_reviews = len(filtered_revlog[filtered_revlog['ease'] > 1])
    retention = (passed_reviews / total_reviews) * 100 if total_reviews > 0 else 0
    col2.metric("True Retention", f"{retention:.1f}%")

known_cards = len(filtered_cards[filtered_cards['knowledge_state'] == 'Known'])
col3.metric("Mature / Known Cards", known_cards)

if not filtered_revlog.empty:
    daily_totals = filtered_revlog.groupby('review_date').size().reset_index()
    daily_totals = daily_totals.sort_values('review_date', ascending=False)

    streak = 0
    check_date = get_local_today()
    if check_date not in daily_totals['review_date'].values:
        check_date = check_date - timedelta(days=1)

    for d in daily_totals['review_date']:
        if d == check_date:
            streak += 1
            check_date = check_date - timedelta(days=1)
        else:
            break
    col4.metric("Current Streak", f"{streak} Days 🔥")

st.divider()

# ==========================================
# DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, mapping, tab7, tab9 = st.tabs([
    "📈 Overview", "🔮 Future Workload", "⏱️ Study Optimization", "🏷️ Difficulty", "🌌 3D Maps", "🎯 Readiness", "Game"
])

# --- TAB 1: OVERVIEW ---
with tab1:

    subtab1, subtab2, subtab3 = st.tabs(["About", "Charts", "Mastery"])

    with subtab1:
        st.markdown("""
        I built this custom dashboard to take a quantitative, data-driven approach to my comprehensive exam prep. Instead of just guessing how well I know the material, this app pulls my raw flashcard data (from Anki) and uses a few data science techniques to measure my actual progress.

        ### 🧠 The Memory Engine: FSRS
        Under the hood, this dashboard uses the **Free Spaced Repetition Scheduler (FSRS)**, a machine-learning algorithm that tracks how memory decays over time. For every single concept I study, the engine calculates:

        * **Stability ($S$):** My *Memory Depth* (in days). This is the algorithm's estimate of how long it will take for my recall probability to drop to 90%. 
        * **Difficulty ($D$):** The *Cognitive Friction* (1-10 scale). This tracks how fundamentally difficult a specific concept is for me to grasp, adjusting automatically based on my failure rates.
        * **Retrievability ($R$):** My *Current Recall Probability* ($R = 0.9^{\\frac{t}{S}}$). The exact likelihood that I would remember the concept if tested today.

        ### 🌌 Mapping the Knowledge (NLP)
        I wanted to be able to visualize my knowledge geographically, so I added a Natural Language Processing pipeline. By applying **TF-IDF vectorization** and **K-Means clustering** to the raw text of my study materials, the app automatically groups related scientific concepts into "semantic islands." This powers the 3D map, letting me visually track which specific domains are stable and which are lagging.

        """)

    with subtab2:
        col_chart1, col_chart2 = st.columns([3, 2])
        with col_chart1:
            st.subheader("Daily Review Volume & Cumulative Total")
            if not filtered_revlog.empty:
                daily_reviews = filtered_revlog.groupby(['review_date', 'ease_label']).size().reset_index(name='count')
                daily_totals = filtered_revlog.groupby('review_date').size().reset_index(name='daily_total').sort_values(
                    'review_date')
                daily_totals['cumulative'] = daily_totals['daily_total'].cumsum()

                fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
                for ease in ['Again', 'Hard', 'Good', 'Easy']:
                    ease_data = daily_reviews[daily_reviews['ease_label'] == ease]
                    if not ease_data.empty:
                        fig_bar.add_trace(go.Bar(x=ease_data['review_date'], y=ease_data['count'], name=ease,
                                                 marker_color=ease_colors[ease]), secondary_y=False)

                fig_bar.add_trace(
                    go.Scatter(x=daily_totals['review_date'], y=daily_totals['cumulative'], name='Cumulative Reviews',
                               mode='lines', line=dict(color='#8b5cf6', width=3)), secondary_y=True)
                fig_bar.update_layout(barmode='stack', hovermode="x unified", margin=dict(t=10))
                fig_bar.update_yaxes(title_text="Daily Reviews", secondary_y=False)
                fig_bar.update_yaxes(title_text="Total Cumulative", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            st.subheader("Knowledge Mastery Breakdown")
            state_counts = filtered_cards['knowledge_state'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']
            fig_pie = px.pie(state_counts, values='Count', names='State', hole=0.4, color='State',
                             color_discrete_map=state_colors)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        col_heat, col_growth, col_create = st.columns(3)
        with col_heat:
            st.subheader("Consistency Heatmap")
            if not filtered_revlog.empty:
                heat_df = filtered_revlog.copy()
                heat_df['review_date'] = pd.to_datetime(heat_df['review_date'])
                daily_heat = heat_df.groupby('review_date').size().reset_index(name='reviews')
                daily_heat['weekday'] = daily_heat['review_date'].dt.weekday
                daily_heat['week_num'] = daily_heat['review_date'].dt.isocalendar().week
                daily_heat['year'] = daily_heat['review_date'].dt.isocalendar().year
                daily_heat['year_week'] = daily_heat['year'].astype(str) + '-W' + daily_heat['week_num'].astype(
                    str).str.zfill(2)

                pivot_heat = daily_heat.pivot_table(index='weekday', columns='year_week', values='reviews', fill_value=0)
                pivot_heat = pivot_heat.reindex([0, 1, 2, 3, 4, 5, 6])
                pivot_heat.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

                fig_heat = px.imshow(pivot_heat, color_continuous_scale="Greens", aspect="auto")
                fig_heat.update_xaxes(showticklabels=False)
                st.plotly_chart(fig_heat, use_container_width=True)

        with col_growth:
            st.subheader("📈 Learning Growth")
            if not filtered_revlog.empty:
                # Calculate when each card was first seen
                first_seen = filtered_revlog.groupby('cid')['review_date'].min().reset_index()
                daily_new_seen = first_seen.groupby('review_date').size().reset_index(name='New Seen')

                fig_seen = px.bar(daily_new_seen, x='review_date', y='New Seen', title="New Cards Seen (First Contact)")
                fig_seen.update_traces(marker_color="#22c55e")
                st.plotly_chart(fig_seen, use_container_width=True)

        with col_create:
            st.subheader("New Cards Added Over Time")
            if not filtered_cards.empty:
                new_cards = filtered_cards.groupby('creation_date').size().reset_index(name='cards_added')
                fig_create = px.line(new_cards, x='creation_date', y='cards_added', markers=True)
                fig_create.update_traces(line_color="#8b5cf6")
                st.plotly_chart(fig_create, use_container_width=True)

    with subtab3:
        with st.expander("📈 Longitudinal Memory Trajectory", expanded=True):
            st.markdown("""
            ### Memory Trajectory & Knowledge Accumulation
            This visualizes how my brain is adapting over time by tracking memory depth and accuracy. 
            * **Cumulative Knowledge (Purple Area):** The total number of unique concepts I've successfully acquired.
            * **7-Day Retention (Green Line):** My smoothed accuracy. If this drops, I am taking on too much new information too quickly.
            * **Memory Depth Expansion (Blue Area):** The average spacing interval of my daily reviews. As this climbs, my knowledge is moving from short-term friction into long-term stable memory.
            """)

            if not filtered_revlog.empty:
                with st.spinner("Calculating longitudinal memory decay and expansion..."):
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go

                    # 1. Cumulative Knowledge (Unique Cards Learned)
                    # Find the first review date for every card to map out when you "acquired" the concept
                    first_seen = filtered_revlog.groupby('cid')['review_date'].min().reset_index()
                    daily_new = first_seen.groupby('review_date').size().reset_index(name='new_cards')
                    daily_new['Cumulative_Knowledge'] = daily_new['new_cards'].cumsum()

                    # 2. Daily Performance & Interval Metrics
                    daily_stats = filtered_revlog.groupby('review_date').agg(
                        total_reviews=('id', 'count'),
                        passed_reviews=('ease', lambda x: (x > 1).sum()),
                        # Only average the interval of cards that actually have an interval > 0 (ignores learning steps)
                        avg_ivl=('ivl', lambda x: x[x > 0].mean())
                    ).reset_index()

                    daily_stats['Retention'] = (daily_stats['passed_reviews'] / daily_stats['total_reviews']) * 100

                    # 3. Merge and Forward Fill
                    # We merge them so we have a continuous timeline, forward filling cumulative knowledge on rest days
                    ts_df = pd.merge(daily_stats, daily_new[['review_date', 'Cumulative_Knowledge']], on='review_date',
                                     how='outer')
                    ts_df = ts_df.sort_values('review_date')
                    ts_df['Cumulative_Knowledge'] = ts_df['Cumulative_Knowledge'].ffill().fillna(0)

                    # 4. Exponential Moving Averages (EMA) to smooth the noise
                    # EMA is better than simple rolling averages here because it reacts faster to recent changes in your study habits
                    ts_df['EMA_Retention'] = ts_df['Retention'].ewm(span=7, adjust=False).mean()
                    ts_df['EMA_Ivl'] = ts_df['avg_ivl'].ewm(span=7, adjust=False).mean()

                    # 5. Build the Dual-Layer Plotly Graph
                    fig_ts = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(
                        "Knowledge Acquisition vs. Retention Accuracy", "Memory Depth Expansion (Avg Interval)"),
                        row_heights=[0.6, 0.4],
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                    )

                    # --- SUBPLOT 1: Knowledge vs Retention ---
                    fig_ts.add_trace(
                        go.Scatter(x=ts_df['review_date'], y=ts_df['Cumulative_Knowledge'],
                                   fill='tozeroy', mode='lines', line=dict(color='rgba(139, 92, 246, 0.6)', width=2),
                                   name="Total Concepts Learned"),
                        row=1, col=1, secondary_y=False
                    )

                    fig_ts.add_trace(
                        go.Scatter(x=ts_df['review_date'], y=ts_df['EMA_Retention'],
                                   mode='lines', line=dict(color='#10b981', width=3),
                                   name="7-Day EMA Retention %"),
                        row=1, col=1, secondary_y=True
                    )

                    # --- SUBPLOT 2: Memory Depth (Intervals) ---
                    fig_ts.add_trace(
                        go.Scatter(x=ts_df['review_date'], y=ts_df['EMA_Ivl'],
                                   fill='tozeroy', mode='lines', line=dict(color='rgba(59, 130, 246, 0.7)', width=3),
                                   name="Avg Memory Depth (Days)"),
                        row=2, col=1
                    )

                    # 6. Layout & Formatting Formatting
                    fig_ts.update_layout(
                        height=650,
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )

                    # Axis stylings
                    fig_ts.update_yaxes(title_text="Unique Cards", showgrid=True, gridcolor='rgba(150,150,150,0.1)',
                                        row=1, col=1, secondary_y=False)
                    fig_ts.update_yaxes(title_text="Retention %", range=[60, 100], showgrid=False, row=1, col=1,
                                        secondary_y=True)
                    fig_ts.update_yaxes(title_text="Days in Memory", showgrid=True, gridcolor='rgba(150,150,150,0.1)',
                                        row=2, col=1)

                    st.plotly_chart(fig_ts, use_container_width=True)

            else:
                st.info("No review logs found to generate the time series.")




# --- TAB 2: FUTURE WORKLOAD & DECAY ---
with tab2:

    subtab1, subtab2 = st.tabs(["Forecasting", "Predictions"])

    with subtab1:
        col_future1, col_future2 = st.columns(2)
        today = get_local_today()

        with col_future1:
            st.subheader("📅 Workload Forecast")
            st.markdown(f"Cards actively scheduled for review by Anki leading up to **{exam_date}**.")
            future_due = filtered_cards[(filtered_cards['due_date'] >= today) & (filtered_cards['due_date'] <= exam_date)]

            if not future_due.empty:
                due_counts = future_due.groupby('due_date').size().reset_index(name='cards_due')
                fig_due = px.bar(due_counts, x='due_date', y='cards_due',
                                 labels={'due_date': 'Date', 'cards_due': 'Cards Due'}, color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_due, use_container_width=True)
            else:
                st.info("No future reviews scheduled in this timeframe.")

        with col_future2:
            st.subheader("📉 Predicted Memory Decay")
            st.markdown("The natural decay curve: when the retention of a card falls below 90% if left unreviewed.")
            future_forgets = filtered_cards[
                (filtered_cards['forgetting_date'] >= today) & (filtered_cards['forgetting_date'] <= exam_date)]

            if not future_forgets.empty:
                decay_counts = future_forgets.groupby('forgetting_date').size().reset_index(name='cards_decaying')
                fig_decay = px.area(decay_counts, x='forgetting_date', y='cards_decaying',
                                    labels={'forgetting_date': 'Date', 'cards_decaying': 'Cards Dropping < 90%'},
                                    color_discrete_sequence=['#ef4444'])
                st.plotly_chart(fig_decay, use_container_width=True)
            else:
                st.info("Not enough FSRS review data to project memory decay.")

        st.divider()
        st.subheader("FSRS Memory State: Difficulty vs. Stability")
        fsrs_plot_df = filtered_cards.dropna(subset=['d', 's', 'card_front'])
        if not fsrs_plot_df.empty:
            fig_scatter = px.scatter(
                fsrs_plot_df, x='d', y='s', hover_data=['clean_text', 'lapses', 'reps'],
                color='deck_name' if selected_deck == "All Decks" else None, opacity=0.6,
                labels={'d': 'Difficulty (1-10)', 's': 'Stability (Days until forgotten)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with subtab2:
        with st.expander("The Neural Oracle", expanded=True):
            st.markdown("""
            ### The Neural Oracle
            This module uses **Dense Semantic Embeddings** and **Gradient Boosting** to predict cognitive friction.
            * **The Regressor:** Predicts the fundamental Difficulty ($D$) of Unseen cards.
            * **The Classifier:** Analyzes cards due soon and predicts my probability of lapsing (failing) based on my historical semantic weaknesses.
            """)


            # 1. Prepare the Core Data
            # We need the embedder you already loaded in Tab 6
            @st.cache_resource
            def load_embedder():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer('all-MiniLM-L6-v2')


            embedder = load_embedder()


            def clean_for_embeddings(text, tags):
                raw = f"{text} {tags}".replace('_', ' ')
                return re.sub(r'[^\w\s]', ' ', str(raw).lower())


            # Safely prepare the dataframe
            ml_df = filtered_cards.copy()
            ml_df['nlp_ready'] = ml_df.apply(lambda x: clean_for_embeddings(x['clean_text'], x['tags']), axis=1)

            if len(ml_df[ml_df['knowledge_state'] != 'Unseen']) > 50:
                with st.spinner("Generating dense 384-D semantic embeddings and training Gradient Boosting trees..."):
                    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
                    import numpy as np

                    # --- MODEL 1: THE FRICTION REGRESSOR (For Unseen Cards) ---
                    st.subheader("🌋 The Horizon: Unseen Difficulty Predictions")

                    # Split data
                    train_reg = ml_df[(ml_df['knowledge_state'] != 'Unseen') & (ml_df['d'].notna())].copy()
                    target_reg = ml_df[ml_df['knowledge_state'] == 'Unseen'].copy()

                    if not target_reg.empty:
                        # Generate Embeddings (X)
                        X_train_emb = embedder.encode(train_reg['nlp_ready'].tolist(), show_progress_bar=False)
                        X_target_emb = embedder.encode(target_reg['nlp_ready'].tolist(), show_progress_bar=False)

                        y_train_reg = train_reg['d']

                        # Train Gradient Boosting Regressor
                        gbr = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
                        gbr.fit(X_train_emb, y_train_reg)

                        # Predict
                        target_reg['Predicted_Difficulty'] = gbr.predict(X_target_emb)

                        # Display Top Threats
                        threats = target_reg[['clean_text', 'deck_name', 'Predicted_Difficulty', 'tags']].sort_values(
                            by='Predicted_Difficulty', ascending=False
                        ).rename(columns={'clean_text':           'Card Text', 'deck_name': 'Deck',
                                          'Predicted_Difficulty': 'Predicted Friction'})

                        st.markdown(
                            "These **Unseen** cards have semantic profiles matching my most difficult historical concepts.")
                        st.dataframe(
                            threats.head(50).style.background_gradient(subset=['Predicted Friction'], cmap='Reds',
                                                                       vmin=1,
                                                                       vmax=10).format(
                                {'Predicted Friction': "{:.1f}"}),
                            use_container_width=True, hide_index=True
                        )
                    else:
                        st.success("No Unseen cards left! The Horizon is clear.")

                    # --- MODEL 2: THE LAPSE CLASSIFIER (For Due Cards) ---
                    st.divider()
                    st.subheader("⚠️ Imminent Lapses: High-Risk Due Cards")

                    # We define a "Struggling" card as one with a historical lapse rate > 15% AND low stability
                    train_clf = ml_df[ml_df['reps'] > 2].copy()
                    train_clf['failure_rate'] = train_clf['lapses'] / train_clf['reps']

                    # Target: 1 if it's a high-risk card, 0 otherwise
                    train_clf['is_high_risk'] = ((train_clf['failure_rate'] > 0.15) & (train_clf['s'] < 14)).astype(int)

                    # We only want to predict on cards that are actually Due in the next 3 days
                    future_3_days = get_local_today() + timedelta(days=3)
                    due_target = ml_df[(ml_df['due_date'].notna()) & (ml_df['due_date'] <= future_3_days) & (
                            ml_df['knowledge_state'] != 'Unseen')].copy()

                    if not due_target.empty and train_clf['is_high_risk'].sum() > 10:
                        # Feature Engineering for Classifier: Embeddings + FSRS State
                        # We stack the dense embeddings with the current FSRS metrics
                        X_clf_emb = embedder.encode(train_clf['nlp_ready'].tolist(), show_progress_bar=False)
                        X_clf_fsrs = train_clf[['d', 's', 'ivl']].fillna(0).values
                        X_train_clf = np.hstack((X_clf_emb, X_clf_fsrs))

                        y_train_clf = train_clf['is_high_risk']

                        # Train Gradient Boosting Classifier
                        gbc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced',
                                                             random_state=42)
                        gbc.fit(X_train_clf, y_train_clf)

                        # Prepare Target Data
                        X_due_emb = embedder.encode(due_target['nlp_ready'].tolist(), show_progress_bar=False)
                        X_due_fsrs = due_target[['d', 's', 'ivl']].fillna(0).values
                        X_target_clf = np.hstack((X_due_emb, X_due_fsrs))

                        # Predict Probability of being a High-Risk Lapse
                        probs = gbc.predict_proba(X_target_clf)[:, 1] * 100
                        due_target['Lapse_Probability'] = probs


                        # Calculate current FSRS Retrievability (R)
                        # Formula: R = 0.9 ^ (t / S)
                        def calc_retrievability(row):
                            if pd.isna(row['last_review_datetime']) or pd.isna(row['s']) or row['s'] <= 0:
                                return 100.0
                            days_since = (get_local_today() - row['last_review_datetime'].date()).days
                            r = (0.9 ** (days_since / row['s'])) * 100
                            return max(0.0, min(100.0, r))


                        due_target['Current_Recall_Prob'] = due_target.apply(calc_retrievability, axis=1)

                        # Filter and sort by the highest machine-learning predicted lapse risk
                        lapses = due_target[due_target['Lapse_Probability'] > 50].sort_values('Lapse_Probability',
                                                                                              ascending=False)

                        if not lapses.empty:
                            st.markdown(
                                f"The Neural Oracle has flagged **{len(lapses)} cards** due soon that have a highly toxic mix of semantic complexity and low stability.")

                            lapse_display = lapses[
                                ['clean_text', 'due_date', 'Lapse_Probability', 'Current_Recall_Prob', 'd']].rename(
                                columns={
                                    'clean_text':        'Card Text', 'due_date': 'Due Date',
                                    'Lapse_Probability': 'AI Lapse Risk (%)', 'Current_Recall_Prob': 'FSRS Recall (%)',
                                    'd':                 'Friction'
                                })

                            st.dataframe(
                                lapse_display.head(50).style.background_gradient(subset=['AI Lapse Risk (%)'],
                                                                                 cmap='Oranges', vmin=50, vmax=100)
                                .background_gradient(subset=['FSRS Recall (%)'], cmap='RdYlGn', vmin=70, vmax=100)
                                .format(
                                    {'AI Lapse Risk (%)': "{:.1f}%", 'FSRS Recall (%)': "{:.1f}%",
                                     'Friction':          "{:.1f}"}),
                                use_container_width=True, hide_index=True
                            )
                        else:
                            st.success("My due cards look remarkably stable. The Oracle predicts no major lapses!")
                    else:
                        st.info("Not enough historical lapse data or upcoming due cards to run the classifier.")

            else:
                st.warning(
                    "I need to review at least 50 cards before the Neural Oracle has enough data to model Lukas' brain.")

# --- TAB 3: STUDY OPTIMIZATION ---
with tab3:
    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        st.subheader("⏰ Retention & Speed by Hour")
        if not filtered_revlog.empty:
            # Calculate Speed (Seconds per card)
            hourly_stats = filtered_revlog.groupby('hour').agg(
                total_reviews=('id', 'count'),
                passed_reviews=('ease', lambda x: (x > 1).sum()),
                avg_time=('time', lambda x: (x.mean() / 1000))  # convert ms to seconds
            ).reset_index()
            hourly_stats['retention'] = (hourly_stats['passed_reviews'] / hourly_stats['total_reviews']) * 100
            hourly_stats = hourly_stats[hourly_stats['total_reviews'] > 10]

            fig_hour = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hour.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['retention'], name="Retention %",
                                          line=dict(color="#10b981", width=3)), secondary_y=False)
            fig_hour.add_trace(go.Bar(x=hourly_stats['hour'], y=hourly_stats['avg_time'], name="Sec/Card",
                                      marker_color="rgba(167, 139, 250, 0.3)"), secondary_y=True)

            fig_hour.update_layout(title="Efficiency vs Accuracy", hovermode="x unified")
            fig_hour.update_yaxes(title_text="Retention %", secondary_y=False, range=[70, 100])
            fig_hour.update_yaxes(title_text="Seconds per Card", secondary_y=True)
            st.plotly_chart(fig_hour, use_container_width=True)

    with col_opt2:
        st.subheader("🎯 Button Bias & Session Stats")
        # Calculate session time estimate
        avg_speed = filtered_revlog['time'].mean() / 1000  # seconds
        cards_today = len(filtered_cards[filtered_cards['due_date'] == get_local_today()])
        est_minutes = (cards_today * avg_speed) / 60

        st.metric("Est. Time to Clear Today", f"{est_minutes:.1f} Mins", help="Based on your historical speed")

        button_counts = filtered_revlog['ease_label'].value_counts().reset_index()
        fig_buttons = px.bar(button_counts, x='ease_label', y='count', color='ease_label',
                             color_discrete_map=ease_colors)
        st.plotly_chart(fig_buttons, use_container_width=True)

    st.divider()
    st.subheader("🗓️ Weekly Retention Heatmap")
    st.markdown("Identifies peak performance hours vs. burnout zones.")

    if not filtered_revlog.empty:
        # 1. Prepare data
        heat_df = filtered_revlog.copy()
        heat_df['weekday'] = pd.to_datetime(heat_df['review_date']).dt.day_name()

        # 2. Calculate retention per day/hour
        # We need a specific order for weekdays
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        heat_stats = heat_df.groupby(['weekday', 'hour']).agg(
            retention=('ease', lambda x: (x > 1).mean() * 100),
            count=('id', 'count')
        ).reset_index()

        # Filter for statistical significance (at least 5 reviews in that slot)
        heat_stats = heat_stats[heat_stats['count'] > 5]

        # 3. Pivot for Heatmap
        pivot_heat = heat_stats.pivot(index='weekday', columns='hour', values='retention')
        pivot_heat = pivot_heat.reindex(days_order)

        # 4. Plot
        fig_heat = px.imshow(
            pivot_heat,
            labels=dict(x="Hour of Day", y="Day of Week", color="Retention %"),
            color_continuous_scale='RdYlGn',  # Red (Fail) to Green (Pass)
            origin='lower',
            aspect="auto"
        )

        fig_heat.update_layout(xaxis_nticks=24)
        st.plotly_chart(fig_heat, use_container_width=True)



# --- TAB 4: TAG ANALYTICS ---
with tab4:
    st.subheader("🏷️ Subject Difficulty by Tag")

    tag_df = filtered_cards.dropna(subset=['d', 'tags']).copy()

    if not tag_df.empty:
        # 1. NEW CLEANING LOGIC: Split by space OR underscore
        # We use a regex [ _] to catch both
        tag_df['tag_list'] = tag_df['tags'].astype(str).str.strip().str.split(r'[ _]')

        exploded_tags = tag_df.explode('tag_list')


        # 2. FILTERING JUNK
        def is_useful_tag(t):
            t = t.lower().strip()
            # Remove empty or tiny fragments
            if len(t) < 3:
                return False
            # Remove paper citations/common noise
            if t in ['moreau', 'payette', 'bauce', 'allen', 'rademacher', 'results', 'discussion', 'background']:
                return False
            # Remove strings that are likely IDs (mix of letters and numbers with length > 5)
            if any(char.isdigit() for char in t) and any(char.isalpha() for char in t) and len(t) > 5:
                return False
            return True


        exploded_tags = exploded_tags[exploded_tags['tag_list'].apply(is_useful_tag)]

        if not exploded_tags.empty:
            tag_stats = exploded_tags.groupby('tag_list').agg(
                avg_difficulty=('d', 'mean'),
                avg_stability=('s', 'mean'),
                card_count=('id', 'count')
            ).reset_index()

            # Show only conceptual tags with at least 5 cards
            tag_stats = tag_stats[tag_stats['card_count'] >= 5]

            # Display hardest tags
            hardest_tags = tag_stats.sort_values(by='avg_difficulty', ascending=True).tail(20)

            if not hardest_tags.empty:
                col_tag_chart, col_tag_data = st.columns([3, 2])
                with col_tag_chart:
                    fig_tags = px.bar(
                        hardest_tags, x='avg_difficulty', y='tag_list',
                        orientation='h', color='avg_difficulty',
                        color_continuous_scale='Reds',
                        title="Top 20 Hardest Concepts (Cleaned)"
                    )
                    fig_tags.update_xaxes(range=[1, 10])
                    st.plotly_chart(fig_tags, use_container_width=True)

                with col_tag_data:
                    st.dataframe(tag_stats.sort_values(by='avg_difficulty', ascending=False), use_container_width=True,
                                 hide_index=True)
            else:
                st.info("No conceptual tags met the filter criteria.")

    st.subheader("Leech & High-Difficulty Cards")
    fsrs_plot_df = filtered_cards.dropna(subset=['d', 's', 'clean_text'])
    if not fsrs_plot_df.empty:
        problem_cards = fsrs_plot_df[
            ['clean_text', 'deck_name', 'knowledge_state', 'd', 's', 'lapses', 'due_date']].sort_values(
            by=['d', 'lapses'], ascending=[False, False])
        problem_cards = problem_cards.rename(
            columns={'clean_text': 'Card Text', 'd': 'Difficulty', 's': 'Stability (Days)',
                     'due_date':   'Anki Due Date'})
        st.dataframe(problem_cards, use_container_width=True, hide_index=True, height=500)





# --- TAB 6: CLEANED 3D t-SNE MAP ---
with mapping:

    subtab_cards, subtab_Meta = st.tabs(["Semantic Knowledge", "Heuristic Strategy"])

    with subtab_cards:
        st.markdown("""
        This map clusters cards based on their **deep semantic meaning** (using AI embeddings).
        * **Nodes (Dots):** Individual flashcards. Color is Difficulty (Red = Hardest).
        * **Islands:** Formed by density. Outliers (Noise) are left unclustered to preserve accuracy.
        * **The Web (Lines):** Connects cards across the space that share a high semantic similarity.
        """)

        # 1. Map & Web Settings
        st.sidebar.divider()
        st.sidebar.header("🌌 Map & Web Settings")
        show_labels = st.sidebar.checkbox("Show Island Labels", value=True)
        min_cluster_size = st.sidebar.slider("Min Cards per Island", 3, 20, 5)
        custom_neighbors = st.sidebar.slider("Map Detail (Neighbors)", 5, 50, 15)
        edge_threshold = st.sidebar.slider("Web Threshold (Similarity)", 0.60, 0.95, 0.75, 0.01)

        # CHANGE 1: Drop only cards without text. Keep Unseen cards!
        map_df = filtered_cards.dropna(subset=['clean_text']).copy()

        # Fill missing FSRS data with 0s so Plotly hover text doesn't crash on Unseen cards
        map_df['d'] = map_df['d'].fillna(0)
        map_df['s'] = map_df['s'].fillna(0)
        map_df['lapses'] = map_df['lapses'].fillna(0)


        # CHANGE 2: Define your 5-Tier Mastery Logic
        def assign_mastery(row):
            state = row.get('knowledge_state', 'Unseen')
            s = row['s']
            d = row['d']
            lapses = row['lapses']

            if state == 'Unseen':
                return 'Not Known'
            elif lapses >= 2 and s < 10:
                return 'Danger'
            elif d > 7.5 or (state == 'Seen' and s < 7):
                return 'Problematic'
            elif s >= 21 or state == 'Known':
                return 'Mastered'
            else:
                return 'Known'


        map_df['Mastery_Zone'] = map_df.apply(assign_mastery, axis=1)

        # Hex codes for your exact requested color scheme
        mastery_colors = {
            'Mastered':    '#14532d',  # Dark Green
            'Known':       '#4ade80',  # Light Green
            'Problematic': '#f97316',  # Orange
            'Danger':      '#ef4444',  # Red
            'Not Known':   '#7f1d1d'  # Deep Red
        }

        if len(map_df) > 30:
            with st.spinner("Generating dense semantic embeddings and mapping the knowledge space..."):

                # 2. Load Local Embedding Model
                @st.cache_resource
                def load_embedder():
                    from sentence_transformers import SentenceTransformer
                    return SentenceTransformer('all-MiniLM-L6-v2')


                embedder = load_embedder()


                # 3. Clean Text (Fixing the Underscore Issue)
                def clean_for_embeddings(text, tags):
                    # Replace underscores with spaces BEFORE the regex so tags are read as distinct words
                    raw = f"{text} {tags}".replace('_', ' ')
                    clean_text = re.sub(r'[^\w\s]', ' ', str(raw).lower())
                    return clean_text


                map_df['nlp_ready'] = map_df.apply(lambda x: clean_for_embeddings(x['clean_text'], x['tags']), axis=1)


                # 4. Generate Dense Embeddings
                @st.cache_data
                def get_embeddings(text_list):
                    return embedder.encode(text_list, show_progress_bar=False)


                embeddings = get_embeddings(map_df['nlp_ready'].tolist())

                import umap
                import hdbscan
                from sklearn.metrics.pairwise import cosine_similarity

                # 5. UMAP Dimensionality Reduction (3D)
                final_neighbors = min(custom_neighbors, len(map_df) - 1)
                reducer = umap.UMAP(
                    n_components=3,
                    n_neighbors=final_neighbors,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                coords = reducer.fit_transform(embeddings)

                map_df['Map X'] = coords[:, 0]
                map_df['Map Y'] = coords[:, 1]
                map_df['Map Z'] = coords[:, 2]

                # 6. HDBSCAN Density-Based Clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                map_df['cluster'] = clusterer.fit_predict(coords)

                # 7. Extract Semantic Labels (Fixing the IndexError & Trigrams)
                cluster_labels = {}
                if show_labels:
                    all_stops = GLOBAL_STOP_WORDS

                    grouped_docs = map_df.groupby('cluster')['nlp_ready'].apply(lambda x: ' '.join(x)).reset_index()

                    # FIX: Filter out noise (-1) AND reset the index so it matches the 0-based SciPy matrix
                    valid_docs = grouped_docs[grouped_docs['cluster'] != -1].reset_index(drop=True)

                    if not valid_docs.empty:
                        # Restored ngram_range=(1, 3) for highly specific medical/scientific labels
                        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 3))
                        tfidf_matrix = vectorizer.fit_transform(valid_docs['nlp_ready'])
                        feature_names = vectorizer.get_feature_names_out()

                        for i, row in valid_docs.iterrows():
                            cluster_id = row['cluster']
                            tfidf_scores = tfidf_matrix[i].toarray()[0]
                            top_indices = tfidf_scores.argsort()[::-1]

                            best_label = f"Concept {cluster_id}"
                            for idx in top_indices:
                                candidate = feature_names[idx]

                                # Safeguard against bad tokens
                                has_stop = any(stop in candidate for stop in all_stops)
                                has_digit = any(char.isdigit() for char in candidate)
                                has_underscore = '_' in candidate

                                if not has_stop and not has_digit and not has_underscore and len(candidate) > 4:
                                    best_label = candidate.upper()
                                    break
                            cluster_labels[cluster_id] = best_label

                # 8. Build the Semantic Web (Edges)
                sim_matrix = cosine_similarity(embeddings)
                edge_x = []
                edge_y = []
                edge_z = []

                # Loop to find connections and build the line trace (upper triangle to avoid duplicates)
                for i in range(len(map_df)):
                    for j in range(i + 1, len(map_df)):
                        if sim_matrix[i, j] > edge_threshold:
                            edge_x.extend([map_df['Map X'].iloc[i], map_df['Map X'].iloc[j], None])
                            edge_y.extend([map_df['Map Y'].iloc[i], map_df['Map Y'].iloc[j], None])
                            edge_z.extend([map_df['Map Z'].iloc[i], map_df['Map Z'].iloc[j], None])

                # 9. Final 3D Plot Assembly
                            # 9. Final 3D Plot Assembly
                            fig_map = px.scatter_3d(
                                map_df, x='Map X', y='Map Y', z='Map Z',
                                color='Mastery_Zone',
                                color_discrete_map=mastery_colors,
                                category_orders={
                                    "Mastery_Zone": ["Mastered", "Known", "Problematic", "Danger", "Not Known"]},
                                # Forces legend order
                                hover_name='deck_name',
                                hover_data={
                                    'Map X':        False, 'Map Y': False, 'Map Z': False,
                                    'clean_text':   True,
                                    'Mastery_Zone': True,
                                    'd':            ':.1f',
                                    's':            ':.1f',
                                    'cluster':      True
                                },
                                opacity=0.9, height=800
                            )

                # Add the Web Layer (Lines)
                fig_map.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='rgba(150, 150, 150, 0.2)', width=1),
                    hoverinfo='none',
                    showlegend=False
                ))

                # Add the Text Labels Layer
                if show_labels:
                    centers = map_df[map_df['cluster'] != -1].groupby('cluster')[
                        ['Map X', 'Map Y', 'Map Z']].mean().reset_index()
                    for _, row in centers.iterrows():
                        label = cluster_labels.get(row['cluster'], "")
                        fig_map.add_trace(go.Scatter3d(
                            x=[row['Map X']], y=[row['Map Y']], z=[row['Map Z']],
                            mode='text',
                            text=[f"<b>{label}</b>"],
                            textfont=dict(color='white', size=14),
                            showlegend=False,
                            hoverinfo='none'
                        ))

                fig_map.update_traces(marker=dict(size=4))
                fig_map.update_layout(scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title='')
                ))
                st.plotly_chart(fig_map, use_container_width=True)

                # 10. Target Acquisition Metrics (Top 3 Hardest Islands)
                st.divider()
                st.subheader("🌋 Target Acquisition: Top 3 Hardest Concept Islands")

                valid_clusters = map_df[map_df['cluster'] != -1]

                if not valid_clusters.empty:
                    island_stats = valid_clusters.groupby('cluster').agg(
                        avg_diff=('d', 'mean'),
                        card_count=('id', 'count')
                    ).reset_index()

                    hardest_islands = island_stats.sort_values(by='avg_diff', ascending=False).head(3)
                    cols = st.columns(len(hardest_islands))

                    for idx, (_, row) in enumerate(hardest_islands.iterrows()):
                        cluster_id = int(row['cluster'])
                        concept_name = cluster_labels.get(cluster_id, f"Concept {cluster_id}")

                        with cols[idx]:
                            st.metric(
                                label=f"🏝️ {concept_name} (Cards: {int(row['card_count'])})",
                                value=f"Friction: {row['avg_diff']:.2f}/10"
                            )
                else:
                    st.info("Not enough clear conceptual islands formed to calculate targets.")

        else:
            st.info("You need at least 30 cards to generate this map.")

    with subtab_Meta:
        st.markdown("""
        ### FSRS Principal Component Analysis (PCA) & Clustering
        This map uses **PCA** to reduce your raw FSRS stats down to 3 core dimensions. 
        Instead of a black-box AI algorithm, PCA creates linear axes that mathematically explain the most variance in your study habits. 
        """)

        # 1. Define FSRS features
        fsrs_features = ['d', 's', 'lapses', 'reps', 'ivl']
        available_f = [f for f in fsrs_features if f in filtered_cards.columns]

        pca_df = filtered_cards.dropna(subset=['d', 's']).copy()

        if len(pca_df) > 15 and len(available_f) >= 3:
            with st.spinner("Running PCA and K-Means on FSRS attributes..."):
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                from sklearn.cluster import KMeans

                # 2. Impute and Scale
                for f in available_f:
                    median_val = pca_df[f].median()
                    pca_df[f] = pca_df[f].fillna(median_val if not pd.isna(median_val) else 0)

                final_pca_df = pca_df.dropna(subset=available_f).copy()
                scaler = StandardScaler()
                fsrs_scaled = scaler.fit_transform(final_pca_df[available_f])

                # 3. Apply PCA (3 Components)
                pca = PCA(n_components=3)
                pca_coords = pca.fit_transform(fsrs_scaled)

                final_pca_df['PC1'] = pca_coords[:, 0]
                final_pca_df['PC2'] = pca_coords[:, 1]
                final_pca_df['PC3'] = pca_coords[:, 2]

                # 4. K-Means Clustering on the Scaled Data
                n_clusters = min(5, len(final_pca_df))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                final_pca_df['Cluster_ID'] = kmeans.fit_predict(fsrs_scaled)

                # 5. Interpret Clusters (Generate Names based on actual centroids)
                # We inverse_transform the centers back to raw FSRS values so they make sense
                centers_raw = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=available_f)
                cluster_names = {}

                avg_lapses = final_pca_df['lapses'].mean()
                avg_reps = final_pca_df['reps'].mean()

                for i, row in centers_raw.iterrows():
                    if row.get('lapses', 0) > max(avg_lapses * 1.5, 2):
                        cluster_names[i] = "🛑 High-Lapse Leeches"
                    elif row.get('d', 0) > 7.5 and row.get('s', 0) < 14:
                        cluster_names[i] = "⚠️ High Friction / Unstable"
                    elif row.get('s', 0) > 30 and row.get('d', 0) < 5:
                        cluster_names[i] = "💎 Mastered & Easy"
                    elif row.get('reps', 0) > avg_reps * 1.5:
                        cluster_names[i] = "🐢 Heavy Grind"
                    else:
                        cluster_names[i] = "⚡ Average / Core"

                final_pca_df['FSRS_Profile'] = final_pca_df['Cluster_ID'].map(cluster_names)

                # 6. Extract PCA Loadings to dynamically name the axes
                loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=available_f)


                def get_axis_label(pc_col):
                    # Find the FSRS metric that pulls the hardest on this axis
                    top_feature = loadings[pc_col].abs().idxmax()
                    direction = "+" if loadings.loc[top_feature, pc_col] > 0 else "-"
                    return f"{pc_col} (Driven by {direction}{top_feature.upper()})"


                ax_x = get_axis_label('PC1')
                ax_y = get_axis_label('PC2')
                ax_z = get_axis_label('PC3')


                # 7. Semantic Extraction for the hover tooltip
                def get_point_concept(text, tags):
                    raw = f"{text} {tags}"
                    text_no_nums = re.sub(r'\d+', '', str(raw).lower())
                    tokens = re.split(r'[ _]', re.sub(r'[^\w\s]', ' ', text_no_nums))
                    clean = [t for t in tokens if len(t) > 3 and t not in GLOBAL_STOP_WORDS]
                    return max(clean, key=len).upper() if clean else "MISC"


                final_pca_df['Concept'] = final_pca_df.apply(lambda x: get_point_concept(x['clean_text'], x['tags']),
                                                             axis=1)

                # 8. Render the 3D PCA Plot
                fig_pca = px.scatter_3d(
                    final_pca_df, x='PC1', y='PC2', z='PC3',
                    color='FSRS_Profile',
                    size='d',  # Scale node size by Difficulty
                    hover_name='Concept',
                    hover_data={
                        'clean_text': True,
                        'deck_name':  True,
                        'd':          ':.1f',
                        's':          ':.1f',
                        'lapses':     True,
                        'reps':       True,
                        'ivl':        True,
                        'PC1':        False, 'PC2': False, 'PC3': False, 'Cluster_ID': False
                    },
                    opacity=0.85, height=800,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                fig_pca.update_layout(
                    scene=dict(
                        xaxis_title=ax_x,
                        yaxis_title=ax_y,
                        zaxis_title=ax_z
                    ),
                    legend=dict(title='FSRS Archetype', yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                # Minimum size to ensure easy cards don't vanish
                fig_pca.update_traces(marker=dict(sizemin=3, line=dict(width=0.5, color='DarkSlateGrey')))

                st.plotly_chart(fig_pca, use_container_width=True)

                st.divider()

                # 9. Explained Variance & Loadings Matrix Output
                col_exp, col_load = st.columns([1, 2])
                with col_exp:
                    st.subheader("📊 Explained Variance")
                    st.markdown("How much of your total learning history is captured by each axis.")
                    var_df = pd.DataFrame({
                        'Principal Component': ['PC1', 'PC2', 'PC3'],
                        'Variance Explained':  pca.explained_variance_ratio_ * 100
                    })
                    fig_var = px.bar(var_df, x='Principal Component', y='Variance Explained', text_auto=':.1f')
                    fig_var.update_traces(textposition='outside')
                    fig_var.update_layout(yaxis_title="% Variance Explained", height=300)
                    st.plotly_chart(fig_var, use_container_width=True)

                with col_load:
                    st.subheader("🧠 Component Loadings")
                    st.markdown(
                        "Values further from zero mean that FSRS attribute strongly pulls the card along that axis.")
                    # Color format the dataframe for easy reading
                    st.dataframe(loadings.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1),
                                 use_container_width=True)

                # 10. Show Raw Cluster Centroids
                st.subheader("📋 FSRS Cluster Profiles (Averages)")
                centers_raw.index = [cluster_names.get(i, f"Cluster {i}") for i in centers_raw.index]
                st.dataframe(centers_raw.sort_values('d', ascending=False).style.format("{:.2f}"),
                             use_container_width=True)

        else:
            st.info("Not enough FSRS parameters to run PCA. Keep studying!")

# --- TAB 7: MASTER READINESS & CALIBRATION ---
# --- TAB 7: MASTER READINESS & CALIBRATION ---
with tab7:
    st.subheader("🎯 Master Readiness & FSRS Calibration")

    col_gauge, col_stab = st.columns([1, 2])

    if not filtered_revlog.empty:
        # 1. Calculate Actual Retention (for the caption)
        actual_passed = (filtered_revlog['ease'] > 1).sum()
        actual_total = len(filtered_revlog)
        actual_retention = (actual_passed / actual_total) * 100 if actual_total > 0 else 0

        # 2. Calculate RMSE (for the gauge)
        # We focus on Review cards (type 1) for the most accurate calibration check
        review_logs = filtered_revlog[filtered_revlog['type'] == 1]

        if not review_logs.empty:
            outcomes = (review_logs['ease'] > 1).astype(int)
            mse = ((outcomes - 0.9) ** 2).mean()  # Compares against 90% target
            rmse_score = (mse ** 0.5) * 100
        else:
            # If no reviews yet, use a simple drift calculation from all cards
            rmse_score = abs(actual_retention - 90.0)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rmse_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "FSRS Calibration RMSE (%)", 'font': {'size': 18}},
                gauge={
                    'axis':    {'range': [0, 100], 'tickwidth': 1},
                    'bar':     {'color': "#60a5fa"},
                    'bgcolor': "white",
                    'steps':   [
                        {'range': [0, 10], 'color': '#22c55e'},  # Healthy
                        {'range': [10, 30], 'color': '#facc15'},  # Drifting
                        {'range': [30, 100], 'color': '#ef4444'}  # Miscalibrated
                    ],
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Now actual_retention is definitely defined!
            st.caption(f"**Current True Retention:** {actual_retention:.1f}%")
            st.caption("Target is 90%. Lower RMSE means your FSRS parameters perfectly match your memory.")

        with col_stab:
            # Stability Distribution
            st.markdown(f"**Average Memory Stability:** {filtered_cards['s'].mean():.1f} days")
            fig_stab = px.histogram(
                filtered_cards, x='s', nbins=50,
                title="Knowledge Depth (Stability Distribution)",
                color_discrete_sequence=['#60a5fa']
            )
            fig_stab.update_layout(xaxis_title="Days until Forgotten", yaxis_title="Number of Cards", height=350)
            st.plotly_chart(fig_stab, use_container_width=True)
    else:
        st.info("The Oracle needs review history to calibrate the FSRS model.")

    # 2. Existing Sunburst Super-Cluster
    st.divider()
    st.subheader("🌌 Global Exam Cluster")

    cluster_df = filtered_cards.dropna(subset=['deck_name', 'knowledge_state', 'd']).copy()

    if not cluster_df.empty:
        # 1. ADD THE MISSING COLUMN
        cluster_df['Exam'] = 'April 14 Exam'

        # 2. Split the deck name hierarchy
        cluster_df['levels'] = cluster_df['deck_name'].str.split(' ➔ ')
        cluster_df['Deck_L1'] = cluster_df['levels'].apply(lambda x: x[0] if len(x) > 0 else None)
        cluster_df['Deck_L2'] = cluster_df['levels'].apply(lambda x: x[1] if len(x) > 1 else None)
        cluster_df['Deck_L3'] = cluster_df['levels'].apply(lambda x: x[2] if len(x) > 2 else None)

        # 3. Dynamic Path Setup
        path_hierarchy = ['Exam', 'Deck_L1', 'Deck_L2', 'Deck_L3', 'knowledge_state']
        # Filter path to only include columns that actually exist and aren't all null
        actual_path = [p for p in path_hierarchy if p in cluster_df.columns and cluster_df[p].notnull().any()]

        # 4. Group and Aggregate
        cluster_stats = cluster_df.groupby(actual_path).agg(
            card_count=('id', 'count'),
            avg_difficulty=('d', 'mean')
        ).reset_index()

        # 5. Create the multi-layer Sunburst
        fig_cluster = px.sunburst(
            cluster_stats,
            path=actual_path,
            values='card_count',
            color='avg_difficulty',
            color_continuous_scale='RdYlGn_r',
            range_color=[1, 10],
            hover_data={'card_count': True, 'avg_difficulty': ':.2f'}
        )

        fig_cluster.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=750)
        st.plotly_chart(fig_cluster, use_container_width=True)



# --- TAB 9: THE ADVENTURER'S GUILD & ARMORY ---
import random

with tab9:
    # 1. OPTIMISTIC UI STATE BUFFER
    # Only pull from Supabase if we haven't loaded the stats into this session yet
    if 'db_stats' not in st.session_state:
        fetched_stats = get_user_stats()
        # Fallback dictionary if the database row is completely empty
        if not fetched_stats:
            fetched_stats = {"spent_gold": 0, "bonus_gold": 0, "inventory": {}, "claimed_bounties": {}}
        st.session_state['db_stats'] = fetched_stats

    db_stats = st.session_state['db_stats']

    # Safely initialize keys to prevent KeyError crashes
    if 'bonus_gold' not in db_stats:
        db_stats['bonus_gold'] = 0
    if 'spent_gold' not in db_stats:
        db_stats['spent_gold'] = 0
    if 'claimed_bounties' not in db_stats:
        db_stats['claimed_bounties'] = {}
    if 'inventory' not in db_stats:
        db_stats['inventory'] = {'display_case': []}

    inventory = db_stats['inventory']
    display_case = inventory.get('display_case', [])

    # Run Calculation Engine with the new tax_debt variable
    total_xp, current_gold, gross_gold, interest, daily_tax, avg_s, bonus_gold, tax_debt = calculate_rpg_state(
        filtered_cards, filtered_revlog, db_stats
    )

    user_level = int((total_xp / 150) ** 0.5) + 1
    fortitude_lvl = int(avg_s / 5)
    retention = (filtered_revlog['ease'] > 1).mean() * 100 if not filtered_revlog.empty else 0
    precision_lvl = int(retention / 10)
    avg_time = filtered_revlog['time'].mean() / 1000 if not filtered_revlog.empty else 0
    celerity_lvl = int(max(0, 100 - (avg_time * 5)) / 10)
    seen_cards = len(filtered_cards[filtered_cards['knowledge_state'] != 'Unseen'])
    total_cards = len(filtered_cards)
    cart_lvl = int((seen_cards / total_cards) * 10) if total_cards > 0 else 0

    subtab_guild, subtab_armory = st.tabs(["⚔️ Guild Hall (Quests)", "🏛️ The Armory (Shop)"])

    # ==========================================
    # SUB-TAB 1: GUILD HALL
    # ==========================================
    with subtab_guild:
        with st.container(border=True):
            st.header(f"🛡️ Scholar Level {user_level}")

            xp_for_current = (user_level - 1) ** 2 * 150
            xp_for_next = user_level ** 2 * 150
            denom = (xp_for_next - xp_for_current)
            lvl_progress = (total_xp - xp_for_current) / denom if denom > 0 else 0
            st.progress(min(max(lvl_progress, 0.0), 1.0),
                        text=f"✨ {total_xp - xp_for_current} / {denom} XP to Level {user_level + 1}")

            col_radar, col_treasury = st.columns([2, 1])

            with col_radar:
                radar_data = pd.DataFrame(dict(
                    r=[fortitude_lvl, precision_lvl, celerity_lvl, cart_lvl, fortitude_lvl],
                    theta=['Fortitude', 'Precision', 'Celerity', 'Cartography', 'Fortitude']
                ))
                fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
                fig_radar.update_traces(fill='toself', line_color='#8b5cf6')
                fig_radar.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_radar, use_container_width=True)

                with col_treasury:
                    st.markdown("### 🏛️ Treasury")
                    st.metric("Current Gold Balance", f"💰 {current_gold}")
                    st.caption(f"**Gross XP Gold:** 🪙 {gross_gold}")
                    st.caption(f"**Quest Bonus Gold:** 💰 {bonus_gold}")
                    st.caption(f"**Daily Interest:** 📈 +{interest}")
                    if tax_debt > 0:
                        st.error(f"⚠️ **Tax Debt:** -{tax_debt} Gold (Clear by earning XP!)")

            # --- THE BOUNTY BOARD (INSTANT PAYOUTS) ---
            with st.container(border=True):
                st.subheader("📜 Daily Bounty Board")

                today_str = str(get_local_today())
                bounties = db_stats['claimed_bounties']

                if bounties.get('date') != today_str:
                    bounties = {"date": today_str, "q1": False, "q2": False, "q3": False}
                    db_stats['claimed_bounties'] = bounties

                cards_due_today = len(filtered_cards[filtered_cards['s'] < 1])
                unseen_remaining = total_cards - seen_cards

                scribe_goal = max(20, min(100, cards_due_today))
                explorer_goal = max(5, min(25, int(unseen_remaining * 0.05)))

                today_revs = filtered_revlog[filtered_revlog['review_date'] == get_local_today()]
                today_count = len(today_revs)
                today_new = len(filtered_cards[filtered_cards['creation_date'] == get_local_today()])


                # OPTIMISTIC UPDATE FUNCTION
                def claim_bounty(q_key, reward):
                    # 1. Update Session State (Instant UI)
                    db_stats['claimed_bounties'][q_key] = True
                    db_stats['bonus_gold'] += reward
                    st.session_state['db_stats'] = db_stats

                    # 2. Sync to Database
                    update_user_stats({
                        "claimed_bounties": db_stats['claimed_bounties'],
                        "bonus_gold":       db_stats['bonus_gold']
                    })

                    st.toast(f"Claimed {reward} Gold!", icon="💰")
                    st.rerun()


            q1, q2, q3 = st.columns(3)
            with q1:
                st.info(f"**The Scribe**\n\nComplete {scribe_goal} Reviews")
                st.progress(min(1.0, today_count / scribe_goal) if scribe_goal > 0 else 1.0)
                if bounties.get('q1'):
                    st.success("✅ Claimed")
                elif today_count >= scribe_goal:
                    if st.button("Claim 💰 20", key="btn_q1"):
                        claim_bounty("q1", 20)
                else:
                    st.write(f"{today_count} / {scribe_goal}")

            with q2:
                st.info(f"**The Explorer**\n\nSee {explorer_goal} New Cards")
                st.progress(min(1.0, today_new / explorer_goal) if explorer_goal > 0 else 1.0)
                if bounties.get('q2'):
                    st.success("✅ Claimed")
                elif today_new >= explorer_goal:
                    if st.button("Claim 💰 30", key="btn_q2"):
                        claim_bounty("q2", 30)
                else:
                    st.write(f"{today_new} / {explorer_goal}")

            with q3:
                st.info(f"**The Perfectionist**\n\nMaintain 90% Accuracy")
                today_acc = (today_revs['ease'] > 1).mean() * 100 if today_count > 0 else 0
                st.progress(min(1.0, today_acc / 90))
                if bounties.get('q3'):
                    st.success("✅ Claimed")
                elif today_acc >= 90 and today_count >= 10:
                    if st.button("Claim 💰 50", key="btn_q3"):
                        claim_bounty("q3", 50)
                else:
                    st.write(f"{today_acc:.1f}% / 90.0%")

        # --- THE DUNGEON ---
        with st.container(border=True):
            st.subheader("👺 The Dungeon: Active Threats")
            fragile_cards = filtered_cards[filtered_cards['s'] < 3]

            if not fragile_cards.empty and 'deck_name' in fragile_cards.columns:
                boss_stats = fragile_cards.groupby('deck_name').agg(
                    avg_stability=('s', 'mean'),
                    count=('id', 'count'),
                    total_difficulty=('d', 'sum')
                )
                bosses = boss_stats[boss_stats['count'] > 10].sort_values('avg_stability')

                if not bosses.empty:
                    target_deck = bosses.index[0]
                    boss_data = bosses.loc[target_deck]
                    boss_hp = int(boss_data['total_difficulty'] * 10)
                    display_name = str(target_deck).split(' ➔ ')[-1].upper()

                    col_b1, col_b2 = st.columns([1, 2])
                    with col_b1:
                        st.error(f"⚠️ **BOSS SPAWNED**\n\n**{display_name} Guardian**")
                        st.metric("Boss HP (Friction)", f"🩸 {boss_hp}")
                    with col_b2:
                        if st.button("⚔️ Generate Slay-List", key="btn_boss", use_container_width=True):
                            boss_cards = filtered_cards[
                                (filtered_cards['deck_name'] == target_deck) & (filtered_cards['s'] < 3)]
                            st.dataframe(boss_cards[['clean_text', 'd', 's']].sort_values('d', ascending=False),
                                         height=200)
                            st.info(
                                f"💡 **Strategy:** Search `\"deck:{target_deck}\" prop:s<3` in Anki and review them!")
                else:
                    st.success("✨ The dungeon is clear.")
            else:
                st.success("✨ The dungeon is clear.")

    # ==========================================
    # SUB-TAB 2: THE ARMORY (DAILY SHOP)
    # ==========================================
    with subtab_armory:
        today_seed = get_local_today().toordinal()
        random.seed(today_seed)


        def get_top_concept_words(df_subset, n=15):
            if df_subset.empty:
                return []
            text = " ".join(df_subset['clean_text'].dropna().astype(str).tolist()).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            words = text.split()
            stops = GLOBAL_STOP_WORDS
            clean_words = [w for w in words if len(w) > 4 and w not in stops]
            if not clean_words:
                return []
            return pd.Series(clean_words).value_counts().head(n).index.tolist()


        weapon_pool = get_top_concept_words(filtered_cards[filtered_cards['d'] < 4])
        armor_pool = get_top_concept_words(filtered_cards[filtered_cards['s'] > 21])
        relic_pool = get_top_concept_words(
            filtered_cards[filtered_cards['knowledge_state'].isin(['Unseen', 'Learning'])])

        w_concept = random.choice(weapon_pool).capitalize() if weapon_pool else "Iron"
        a_concept = random.choice(armor_pool).capitalize() if armor_pool else "Leather"
        r_concept = random.choice(relic_pool).capitalize() if relic_pool else "Novice"

        if celerity_lvl > precision_lvl:
            weapon_item = f"🗡️ The {w_concept} Scalpel"
        elif precision_lvl > 7:
            weapon_item = f"🏹 {w_concept} Longbow"
        else:
            weapon_item = f"📖 {w_concept} Grimoire"

        if fortitude_lvl >= 10:
            armor_item = f"🛡️ {a_concept} Carapace"
        elif fortitude_lvl >= 5:
            armor_item = f"🧥 {a_concept} Mantle"
        else:
            armor_item = f"👕 {a_concept} Tunic"

        if cart_lvl >= 9:
            relic_item = f"🧭 The {r_concept} Astrolabe"
        else:
            relic_item = f"🕯️ {r_concept} Lantern"

        with st.container(border=True):
            st.subheader(f"🛒 The Daily Merchant ({get_local_today().strftime('%b %d')})")
            st.markdown(f"**Available Gold:** 💰 {current_gold}")

            col_s1, col_s2, col_s3 = st.columns(3)


            def buy_item(item_name, cost):
                # 1. Update Session State (Instant UI)
                db_stats['spent_gold'] += cost
                display_case.append(item_name)
                db_stats['inventory']['display_case'] = display_case
                st.session_state['db_stats'] = db_stats

                # 2. Sync to Database
                update_user_stats({
                    "spent_gold": db_stats['spent_gold'],
                    "inventory":  db_stats['inventory']
                })

                st.toast(f"Purchased {item_name}!", icon="🎉")
                st.rerun()


            with col_s1:
                st.info(f"**Main Hand**\n\n{weapon_item}")
                if weapon_item in display_case:
                    st.button("Sold Out", disabled=True, key="so_w", use_container_width=True)
                elif current_gold >= 50:
                    if st.button("Buy for 💰 50", key="buy_w", use_container_width=True):
                        buy_item(weapon_item, 50)
                else:
                    st.button("Needs 💰 50", disabled=True, key="poor_w", use_container_width=True)

            with col_s2:
                st.info(f"**Body Armor**\n\n{armor_item}")
                if armor_item in display_case:
                    st.button("Sold Out", disabled=True, key="so_a", use_container_width=True)
                elif current_gold >= 100:
                    if st.button("Buy for 💰 100", key="buy_a", use_container_width=True):
                        buy_item(armor_item, 100)
                else:
                    st.button("Needs 💰 100", disabled=True, key="poor_a", use_container_width=True)

            with col_s3:
                st.info(f"**Accessory**\n\n{relic_item}")
                if relic_item in display_case:
                    st.button("Sold Out", disabled=True, key="so_r", use_container_width=True)
                elif current_gold >= 75:
                    if st.button("Buy for 💰 75", key="buy_r", use_container_width=True):
                        buy_item(relic_item, 75)
                else:
                    st.button("Needs 💰 75", disabled=True, key="poor_r", use_container_width=True)

        with st.container(border=True):
            st.subheader("🏛️ The Grand Hall of Relics")
            st.markdown("Your historical collection of academic artifacts.")

            if display_case:
                weapons = [item for item in display_case if any(icon in item for icon in ['🗡️', '🏹', '📖'])]
                armors = [item for item in display_case if any(icon in item for icon in ['🛡️', '🧥', '👕'])]
                relics = [item for item in display_case if any(icon in item for icon in ['🧭', '🕯️'])]

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**🗡️ Weapons**")
                    for w in reversed(weapons):
                        st.caption(w)
                with c2:
                    st.markdown("**🛡️ Armor**")
                    for a in reversed(armors):
                        st.caption(a)
                with c3:
                    st.markdown("**🧭 Relics**")
                    for r in reversed(relics):
                        st.caption(r)
            else:
                st.info("The hall is empty. Check the Bounty Board to earn your first gold!")


