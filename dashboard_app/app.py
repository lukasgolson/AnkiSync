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

map_df = None

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
tab1, tab2, tab3, tab4, mapping, tab7, tab8 = st.tabs([
    "📈 Overview", "🔮 Future Workload", "⏱️ Study Optimization", "🏷️ Difficulty", "🌌 3D Maps", "🎯 Readiness", "Filter Generator"])

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
            ### The Neural Oracle (V2)
            This module uses **Dense Semantic Embeddings** and **Structural Analytics** to predict cognitive friction.
            * **The Friction Regressor:** Predicts the fundamental Difficulty (1-10) of Unseen cards.
            * **The Time Oracle:** Predicts the cognitive load (Seconds per Review) of Unseen cards.
            * **The Cursed Concept Classifier:** Analyzes cards due soon to flag historically toxic semantic patterns, regardless of what the spaced repetition algorithm says.
            """)


            @st.cache_resource
            def load_embedder():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer('all-MiniLM-L6-v2')


            embedder = load_embedder()


            def clean_for_embeddings(text, tags):
                raw = f"{text} {tags}".replace('_', ' ')
                return re.sub(r'[^\w\s]', ' ', str(raw).lower())


            # --- NEW: Structural Feature Engineering ---
            def get_structural_features(text):
                text_str = str(text)
                word_count = len(text_str.split())
                char_count = len(text_str)
                # Check for Anki cloze deletions (e.g., {{c1::answer}})
                has_cloze = 1 if 'c1::' in text_str or 'c2::' in text_str else 0
                return word_count, char_count, has_cloze


            ml_df = filtered_cards.copy()
            ml_df['nlp_ready'] = ml_df.apply(lambda x: clean_for_embeddings(x['clean_text'], x['tags']), axis=1)

            # Apply structural features
            ml_df[['word_count', 'char_count', 'has_cloze']] = ml_df['card_front'].apply(
                lambda x: pd.Series(get_structural_features(x))
            )

            # --- NEW: Calculate True Target for Learning Time ---
            if not filtered_revlog.empty:
                time_stats = filtered_revlog.groupby('cid')['time'].mean().reset_index()
                # Convert milliseconds to seconds
                time_stats['avg_time_sec'] = time_stats['time'] / 1000.0
                ml_df = ml_df.merge(time_stats[['cid', 'avg_time_sec']], left_on='id', right_on='cid', how='left')

            if len(ml_df[ml_df['knowledge_state'] != 'Unseen']) > 50:
                with st.spinner("Generating 384-D semantic embeddings and training Gradient Boosting trees..."):
                    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
                    import numpy as np

                    # ==========================================
                    # MODEL 1 & 2: THE HORIZON (Difficulty & Time)
                    # ==========================================
                    st.subheader("🌋 The Horizon: Unseen Card Predictions")

                    # Training Data for Regressors
                    train_reg = ml_df[(ml_df['knowledge_state'] != 'Unseen') & (ml_df['d'].notna()) & (
                        ml_df['avg_time_sec'].notna())].copy()
                    target_reg = ml_df[ml_df['knowledge_state'] == 'Unseen'].copy()

                    if not target_reg.empty and not train_reg.empty:
                        # 1. Feature Stacking (Embeddings + Structure)
                        X_train_emb = embedder.encode(train_reg['nlp_ready'].tolist(), show_progress_bar=False)
                        X_train_struct = train_reg[['word_count', 'char_count', 'has_cloze']].values
                        X_train_combined = np.hstack((X_train_emb, X_train_struct))

                        X_target_emb = embedder.encode(target_reg['nlp_ready'].tolist(), show_progress_bar=False)
                        X_target_struct = target_reg[['word_count', 'char_count', 'has_cloze']].values
                        X_target_combined = np.hstack((X_target_emb, X_target_struct))

                        # 2. Train Difficulty Regressor
                        y_train_diff = train_reg['d']
                        gbr_diff = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
                        gbr_diff.fit(X_train_combined, y_train_diff)
                        target_reg['Predicted_Difficulty'] = gbr_diff.predict(X_target_combined)

                        # 3. Train Time/Effort Regressor
                        # Filter out crazy outliers (e.g., leaving Anki open for 30 mins)
                        train_time_clean = train_reg[train_reg['avg_time_sec'] < 120]
                        X_train_time_emb = embedder.encode(train_time_clean['nlp_ready'].tolist(),
                                                           show_progress_bar=False)
                        X_train_time_struct = train_time_clean[['word_count', 'char_count', 'has_cloze']].values
                        X_train_time_combined = np.hstack((X_train_time_emb, X_train_time_struct))

                        y_train_time = train_time_clean['avg_time_sec']
                        gbr_time = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
                        gbr_time.fit(X_train_time_combined, y_train_time)
                        target_reg['Predicted_Time_Sec'] = gbr_time.predict(X_target_combined)

                        # 4. Display Horizon Output
                        threats = target_reg[['clean_text', 'deck_name', 'Predicted_Difficulty', 'Predicted_Time_Sec',
                                              'word_count']].sort_values(
                            by='Predicted_Difficulty', ascending=False
                        ).rename(columns={
                            'clean_text':           'Card Text',
                            'deck_name':            'Deck',
                            'Predicted_Difficulty': 'Predicted Friction (1-10)',
                            'Predicted_Time_Sec':   'Est. Time/Rep (Sec)',
                            'word_count':           'Words'
                        })

                        st.markdown(
                            "These **Unseen** cards have semantic and structural profiles matching your hardest historical concepts.")
                        st.dataframe(
                            threats.head(50).style.background_gradient(subset=['Predicted Friction (1-10)'],
                                                                       cmap='Reds', vmin=1, vmax=10)
                            .background_gradient(subset=['Est. Time/Rep (Sec)'], cmap='Purples', vmin=5, vmax=30)
                            .format({'Predicted Friction (1-10)': "{:.1f}", 'Est. Time/Rep (Sec)': "{:.1f}s"}),
                            use_container_width=True, hide_index=True
                        )
                    else:
                        st.success("No Unseen cards left! The Horizon is clear.")

                    # ==========================================
                    # MODEL 3: THE CURSED CONCEPT CLASSIFIER
                    # ==========================================
                    st.divider()
                    st.subheader("⚠️ Cursed Concepts: High-Risk Semantic Patterns")

                    # Target: Strictly historical failure rate (No FSRS target leakage)
                    train_clf = ml_df[ml_df['reps'] > 3].copy()
                    train_clf['failure_rate'] = train_clf['lapses'] / train_clf['reps']
                    train_clf['is_cursed'] = (train_clf['failure_rate'] > 0.15).astype(int)

                    future_3_days = get_local_today() + timedelta(days=3)
                    due_target = ml_df[(ml_df['due_date'].notna()) & (ml_df['due_date'] <= future_3_days) & (
                                ml_df['knowledge_state'] != 'Unseen')].copy()

                    if not due_target.empty and train_clf['is_cursed'].sum() > 10:

                        # Features: ONLY Embeddings + Structure. Absolutely zero FSRS state data.
                        X_clf_emb = embedder.encode(train_clf['nlp_ready'].tolist(), show_progress_bar=False)
                        X_clf_struct = train_clf[['word_count', 'char_count', 'has_cloze']].values
                        X_train_clf = np.hstack((X_clf_emb, X_clf_struct))

                        y_train_clf = train_clf['is_cursed']

                        # Train Classifier
                        gbc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced',
                                                             random_state=42)
                        gbc.fit(X_train_clf, y_train_clf)

                        # Predict on Due Cards
                        X_due_emb = embedder.encode(due_target['nlp_ready'].tolist(), show_progress_bar=False)
                        X_due_struct = due_target[['word_count', 'char_count', 'has_cloze']].values
                        X_target_clf = np.hstack((X_due_emb, X_due_struct))

                        probs = gbc.predict_proba(X_target_clf)[:, 1] * 100
                        due_target['Cursed_Probability'] = probs


                        # Calculate current FSRS Retrievability (R) as a baseline comparison
                        def calc_retrievability(row):
                            if pd.isna(row['last_review_datetime']) or pd.isna(row['s']) or row['s'] <= 0:
                                return 100.0
                            days_since = (get_local_today() - row['last_review_datetime'].date()).days
                            r = (0.9 ** (days_since / row['s'])) * 100
                            return max(0.0, min(100.0, r))


                        due_target['Current_Recall_Prob'] = due_target.apply(calc_retrievability, axis=1)

                        lapses = due_target[due_target['Cursed_Probability'] > 50].sort_values('Cursed_Probability',
                                                                                               ascending=False)

                        if not lapses.empty:
                            st.markdown(
                                f"The Oracle has flagged **{len(lapses)} upcoming cards** as 'Cursed'. Regardless of what the spacing algorithm says, the actual text of these cards triggers a high historical failure rate for you.")

                            lapse_display = lapses[
                                ['clean_text', 'due_date', 'Cursed_Probability', 'Current_Recall_Prob']].rename(
                                columns={
                                    'clean_text':          'Card Text',
                                    'due_date':            'Due Date',
                                    'Cursed_Probability':  'Semantic Threat Level (%)',
                                    'Current_Recall_Prob': 'FSRS Current Recall (%)'
                                })

                            st.dataframe(
                                lapse_display.head(50).style.background_gradient(subset=['Semantic Threat Level (%)'],
                                                                                 cmap='Oranges', vmin=50, vmax=100)
                                .background_gradient(subset=['FSRS Current Recall (%)'], cmap='RdYlGn', vmin=70,
                                                     vmax=100)
                                .format({'Semantic Threat Level (%)': "{:.1f}%", 'FSRS Current Recall (%)': "{:.1f}%"}),
                                use_container_width=True, hide_index=True
                            )
                        else:
                            st.success("Your upcoming cards are semantically safe. No cursed concepts detected!")
                    else:
                        st.info("Not enough historical lapse data to define 'Cursed' patterns yet.")
            else:
                st.warning("I need more active cards in the rotation before the Oracle can calibrate to your brain.")

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

    st.divider()
    st.subheader("🧭 Knowledge Gaps (Under-Explored Domains)")
    st.markdown(
        "Tags with a high volume of cards, but a large percentage remaining **Unseen** or with very few reviews. These are your blind spots.")

    # 1. Create a NEW dataframe that doesn't drop 'd' (so we keep Unseen cards)
    explore_df = filtered_cards.dropna(subset=['tags']).copy()

    if not explore_df.empty:
        # Split tags just like before
        explore_df['tag_list'] = explore_df['tags'].astype(str).str.strip().str.split(r'[ _]')
        exploded_explore = explore_df.explode('tag_list')

        # Re-use your filter function
        exploded_explore = exploded_explore[exploded_explore['tag_list'].apply(is_useful_tag)]

        if not exploded_explore.empty:
            # 2. Calculate Exploration Metrics
            exploration_stats = exploded_explore.groupby('tag_list').agg(
                total_cards=('id', 'count'),
                unseen_cards=('knowledge_state', lambda x: (x == 'Unseen').sum()),
                avg_reps=('reps', lambda x: x.fillna(0).mean())  # Unseen cards have NaN reps
            ).reset_index()

            # 3. Calculate % Unseen and a "Priority Score"
            # Priority Score = Volume of Unseen Cards * Percentage of Unseen
            # This surfaces tags that have BOTH a lot of hidden cards AND are heavily ignored
            exploration_stats['unseen_pct'] = (exploration_stats['unseen_cards'] / exploration_stats['total_cards']) * 100

            blind_spots = exploration_stats[
                (exploration_stats['total_cards'] >= 5) &
                (exploration_stats['unseen_cards'] > 0)
                ].copy()

            if not blind_spots.empty:
                blind_spots['priority_score'] = blind_spots['unseen_cards'] * blind_spots['unseen_pct']
                top_blind_spots = blind_spots.sort_values(by='priority_score', ascending=False).head(15)

                col_gap_chart, col_gap_data = st.columns([3, 2])

                with col_gap_chart:
                    # Scatter plot showing Volume vs. Unexplored Percentage
                    fig_gaps = px.scatter(
                        top_blind_spots,
                        x='total_cards',
                        y='unseen_pct',
                        size='unseen_cards',
                        color='unseen_pct',
                        hover_name='tag_list',
                        text='tag_list',
                        color_continuous_scale='Purples',
                        labels={'total_cards': 'Total Cards in Concept', 'unseen_pct': '% Unseen'},
                        title="Blind Spot Matrix"
                    )
                    fig_gaps.update_traces(textposition='top center')
                    fig_gaps.update_layout(yaxis=dict(range=[0, 110]))  # Give labels room to breathe
                    st.plotly_chart(fig_gaps, use_container_width=True)

                with col_gap_data:
                    display_gaps = top_blind_spots[['tag_list', 'total_cards', 'unseen_cards', 'unseen_pct']].rename(
                        columns={
                            'tag_list':     'Concept Tag',
                            'total_cards':  'Total',
                            'unseen_cards': 'Unseen',
                            'unseen_pct':   '% Unseen'
                        })

                    st.dataframe(
                        display_gaps.style.background_gradient(subset=['% Unseen'], cmap='Purples', vmin=0, vmax=100)
                        .format({'% Unseen': "{:.1f}%"}),
                        use_container_width=True, hide_index=True
                    )




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
        map_dimension = st.sidebar.radio("Map Dimension", ["2D", "3D"], index=0, horizontal=True)
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
            with st.spinner(f"Generating dense semantic embeddings and mapping the {map_dimension} knowledge space..."):

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

                # 5. UMAP Dimensionality Reduction (Dynamic 2D or 3D)
                final_neighbors = min(custom_neighbors, len(map_df) - 1)
                n_comp = 2 if map_dimension == "2D" else 3

                reducer = umap.UMAP(
                    n_components=n_comp,
                    n_neighbors=final_neighbors,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                coords = reducer.fit_transform(embeddings)

                map_df['Map X'] = coords[:, 0]
                map_df['Map Y'] = coords[:, 1]
                if map_dimension == "3D":
                    map_df['Map Z'] = coords[:, 2]

                # 6. HDBSCAN Density-Based Clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                map_df['cluster'] = clusterer.fit_predict(coords)

                # 7. Extract Semantic Labels
                cluster_labels = {}
                if show_labels:
                    all_stops = GLOBAL_STOP_WORDS

                    grouped_docs = map_df.groupby('cluster')['nlp_ready'].apply(lambda x: ' '.join(x)).reset_index()
                    valid_docs = grouped_docs[grouped_docs['cluster'] != -1].reset_index(drop=True)

                    if not valid_docs.empty:
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
                                has_stop = any(stop in candidate for stop in all_stops)
                                has_digit = any(char.isdigit() for char in candidate)
                                has_underscore = '_' in candidate

                                if not has_stop and not has_digit and not has_underscore and len(candidate) > 4:
                                    best_label = candidate.upper()
                                    break
                            cluster_labels[cluster_id] = best_label

                # 8. Build the Semantic Web (Edges)
                sim_matrix = cosine_similarity(embeddings)
                edge_x, edge_y, edge_z = [], [], []

                for i in range(len(map_df)):
                    for j in range(i + 1, len(map_df)):
                        if sim_matrix[i, j] > edge_threshold:
                            edge_x.extend([map_df['Map X'].iloc[i], map_df['Map X'].iloc[j], None])
                            edge_y.extend([map_df['Map Y'].iloc[i], map_df['Map Y'].iloc[j], None])
                            if map_dimension == "3D":
                                edge_z.extend([map_df['Map Z'].iloc[i], map_df['Map Z'].iloc[j], None])

                # 9. Final Plot Assembly
                                # 9. Final Plot Assembly
                if map_dimension == "2D":
                    fig_map = px.scatter(
                        map_df, x='Map X', y='Map Y',
                        color='Mastery_Zone', color_discrete_map=mastery_colors,
                        category_orders={"Mastery_Zone": ["Mastered", "Known", "Problematic", "Danger",
                                                          "Not Known"]},
                        hover_name='deck_name',
                        hover_data={'Map X':        False, 'Map Y': False, 'clean_text': True,
                                    'Mastery_Zone': True, 'd': ':.1f', 's': ':.1f', 'cluster': True},
                        opacity=0.9, height=800
                    )

                    # Add the Web Layer (Lines)
                    fig_map.add_trace(go.Scatter(
                        x=edge_x, y=edge_y, mode='lines',
                        line=dict(color='rgba(150, 150, 150, 0.2)', width=1),
                        hoverinfo='skip', showlegend=False  # Changed to skip
                    ))

                    # LAYER HACK: Move the newly added line trace to the bottom of the stack
                    fig_map.data = (fig_map.data[-1],) + fig_map.data[:-1]

                    # Add the Text Labels Layer (On top, but ghosted to mouse)
                    if show_labels:
                        centers = map_df[map_df['cluster'] != -1].groupby('cluster')[
                            ['Map X', 'Map Y']].mean().reset_index()
                        for _, row in centers.iterrows():
                            label = cluster_labels.get(row['cluster'], "")
                            fig_map.add_trace(go.Scatter(
                                x=[row['Map X']], y=[row['Map Y']], mode='text',
                                text=[f"<b>{label}</b>"], textfont=dict(color='white', size=14),
                                showlegend=False, hoverinfo='skip'  # Changed to skip
                            ))

                    # Target only markers to avoid altering lines/text
                    fig_map.update_traces(
                        marker=dict(size=6, line=dict(width=0.5, color='rgba(0,0,0,0.5)')),
                        selector=dict(mode='markers')
                    )

                    fig_map.update_layout(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )

                else:  # 3D rendering
                    fig_map = px.scatter_3d(
                        map_df, x='Map X', y='Map Y', z='Map Z',
                        color='Mastery_Zone', color_discrete_map=mastery_colors,
                        category_orders={"Mastery_Zone": ["Mastered", "Known", "Problematic", "Danger",
                                                          "Not Known"]},
                        hover_name='deck_name',
                        hover_data={'Map X':        False, 'Map Y': False, 'Map Z': False,
                                    'clean_text':   True, 'Mastery_Zone': True, 'd': ':.1f',
                                    's':            ':.1f', 'cluster': True},
                        opacity=0.9, height=800
                    )

                    # Add the Web Layer
                    fig_map.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z, mode='lines',
                        line=dict(color='rgba(150, 150, 150, 0.2)', width=1),
                        hoverinfo='skip', showlegend=False
                    ))

                    # LAYER HACK: Move the lines to the bottom of the stack
                    fig_map.data = (fig_map.data[-1],) + fig_map.data[:-1]

                    if show_labels:
                        centers = map_df[map_df['cluster'] != -1].groupby('cluster')[
                            ['Map X', 'Map Y', 'Map Z']].mean().reset_index()
                        for _, row in centers.iterrows():
                            label = cluster_labels.get(row['cluster'], "")
                            fig_map.add_trace(go.Scatter3d(
                                x=[row['Map X']], y=[row['Map Y']], z=[row['Map Z']], mode='text',
                                text=[f"<b>{label}</b>"], textfont=dict(color='white', size=14),
                                showlegend=False, hoverinfo='skip'
                            ))

                    # Target only markers
                    fig_map.update_traces(marker=dict(size=4), selector=dict(mode='markers'))

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
        ### Explicit FSRS Heuristic Space
        This map plots cards directly across the three core dimensions of spaced repetition memory.
        * **X-Axis (Memory Depth):** Stability (Log-scaled). Cards moving right are solidifying into long-term memory.
        * **Y-Axis (Cognitive Friction):** FSRS Difficulty (1-10). Cards moving up are fundamentally harder concepts.
        * **Z-Axis (Struggle Rate):** Historical failure rate (Lapses ÷ Reps). 
        """)

        # 1. Define FSRS features
        fsrs_features = ['d', 's', 'lapses', 'reps', 'ivl']
        available_f = [f for f in fsrs_features if f in filtered_cards.columns]

        pca_df = filtered_cards.dropna(subset=['d', 's']).copy()

        if len(pca_df) > 15 and len(available_f) >= 3:
            with st.spinner(f"Mapping explicit FSRS dimensions in {map_dimension}..."):
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                import numpy as np

                # 2. Impute missing values
                for f in available_f:
                    median_val = pca_df[f].median()
                    pca_df[f] = pca_df[f].fillna(median_val if not pd.isna(median_val) else 0)

                final_pca_df = pca_df.dropna(subset=available_f).copy()

                # 3. Feature Engineering for Explicit Axes
                final_pca_df['s_plot'] = final_pca_df['s'].clip(lower=0.1)

                final_pca_df['lapse_rate'] = np.where(
                    final_pca_df['reps'] > 0,
                    (final_pca_df['lapses'] / final_pca_df['reps']) * 100,
                    0
                )

                final_pca_df['marker_size'] = final_pca_df['reps'].clip(lower=3, upper=50)

                # --- NEW: VISUAL JITTER ---
                # We add random noise to the plot coordinates to break up the "shelves",
                # but keep the exact values for the hover tooltips.
                final_pca_df['d_plot'] = final_pca_df['d'] + np.random.normal(0, 0.15, size=len(final_pca_df))
                final_pca_df['lapse_rate_plot'] = final_pca_df['lapse_rate'] + np.random.normal(0, 1.5,
                                                                                                size=len(final_pca_df))

                # 4. K-Means Clustering
                skewed_features = [f for f in ['s', 'ivl', 'reps', 'lapses'] if f in available_f]
                for f in skewed_features:
                    final_pca_df[f"{f}_log"] = np.log1p(final_pca_df[f])

                features_for_scaling = [f for f in ['d'] if f in available_f] + [f"{f}_log" for f in skewed_features]

                scaler = StandardScaler()
                fsrs_scaled = scaler.fit_transform(final_pca_df[features_for_scaling])

                n_clusters = min(5, len(final_pca_df))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                final_pca_df['Cluster_ID'] = kmeans.fit_predict(fsrs_scaled)

                # 5. Interpret Clusters
                centers_scaled = kmeans.cluster_centers_
                centers_log = scaler.inverse_transform(centers_scaled)
                centers_raw = pd.DataFrame(centers_log, columns=features_for_scaling)

                for f in skewed_features:
                    centers_raw[f] = np.expm1(centers_raw[f"{f}_log"])

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


                # 6. Semantic Extraction for the hover tooltip
                def get_point_concept(text, tags):
                    raw = f"{text} {tags}"
                    text_no_nums = re.sub(r'\d+', '', str(raw).lower())
                    tokens = re.split(r'[ _]', re.sub(r'[^\w\s]', ' ', text_no_nums))
                    clean = [t for t in tokens if len(t) > 3 and t not in GLOBAL_STOP_WORDS]
                    return max(clean, key=len).upper() if clean else "MISC"


                final_pca_df['Concept'] = final_pca_df.apply(lambda x: get_point_concept(x['clean_text'], x['tags']),
                                                             axis=1)

                # 7. Render the Explicit Domain Map
                if map_dimension == "2D":
                    fig_meta = px.scatter(
                        final_pca_df,
                        x='s_plot',
                        y='d_plot',  # Using jittered Y
                        color='FSRS_Profile',
                        size='marker_size',
                        hover_name='Concept',
                        hover_data={
                            'clean_text':  True, 'deck_name': True,
                            'd':           ':.1f',  # Hover shows REAL difficulty
                            's':           ':.1f',
                            'lapses':      True, 'reps': True,
                            'lapse_rate':  ':.1f%',  # Hover shows REAL lapse rate
                            's_plot':      False, 'd_plot': False, 'lapse_rate_plot': False,
                            'marker_size': False, 'Cluster_ID': False
                        },
                        opacity=0.7, height=800,  # Lowered opacity slightly to see density
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )

                    fig_meta.update_layout(
                        xaxis_title="Memory Depth (Stability in Days)",
                        yaxis_title="Cognitive Friction (Difficulty 1-10)",
                        legend=dict(title='FSRS Archetype', yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    fig_meta.update_xaxes(type="log", tickvals=[0.1, 1, 3, 7, 21, 90, 365],
                                          ticktext=["New", "1d", "3d", "1w", "3w", "3m", "1y"])
                    fig_meta.update_traces(marker=dict(line=dict(width=0.5, color='rgba(0,0,0,0.5)')))

                else:  # 3D rendering
                    fig_meta = px.scatter_3d(
                        final_pca_df,
                        x='s_plot',
                        y='d_plot',  # Using jittered Y
                        z='lapse_rate_plot',  # Using jittered Z
                        color='FSRS_Profile',
                        size='marker_size',
                        hover_name='Concept',
                        hover_data={
                            'clean_text':  True, 'deck_name': True,
                            'd':           ':.1f',  # Hover shows REAL difficulty
                            's':           ':.1f',
                            'lapses':      True, 'reps': True,
                            'lapse_rate':  ':.1f%',  # Hover shows REAL lapse rate
                            's_plot':      False, 'd_plot': False, 'lapse_rate_plot': False,
                            'marker_size': False, 'Cluster_ID': False
                        },
                        opacity=0.7, height=800,  # Lowered opacity slightly to see density
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )

                    fig_meta.update_layout(
                        scene=dict(
                            xaxis=dict(type="log", title="Memory Depth (Stability)", tickvals=[0.1, 1, 7, 30, 365],
                                       ticktext=["New", "1d", "1w", "1m", "1y"]),
                            yaxis_title="Cognitive Friction (Difficulty 1-10)",
                            zaxis_title="Struggle Rate (% Lapses/Reps)"
                        ),
                        legend=dict(title='FSRS Archetype', yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    fig_meta.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))

                st.plotly_chart(fig_meta, use_container_width=True)

                st.divider()

                # 8. Show Raw Cluster Centroids
                st.subheader("📋 FSRS Cluster Profiles (Averages)")

                cols_to_drop = [f"{f}_log" for f in skewed_features if f"{f}_log" in centers_raw.columns]
                display_centers = centers_raw.drop(columns=cols_to_drop).copy()
                display_centers.index = [cluster_names.get(i, f"Cluster {i}") for i in display_centers.index]

                st.dataframe(display_centers.sort_values('d', ascending=False).style.format("{:.2f}"),
                             use_container_width=True)

        else:
            st.info("Not enough FSRS parameters to run the mapping. Keep studying!")

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

        # 2. Split using YOUR original arrow delimiter!
        # Make sure the arrow character and spaces exactly match your dataframe
        cluster_df['levels'] = cluster_df['deck_name'].str.split(' ➔ ')

        # 3. Use '(Direct)' to balance the tree, and .strip() to clean up spaces
        cluster_df['Deck_L1'] = cluster_df['levels'].apply(
            lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown')
        cluster_df['Deck_L2'] = cluster_df['levels'].apply(
            lambda x: x[1].strip() if isinstance(x, list) and len(x) > 1 else '(Direct)')
        cluster_df['Deck_L3'] = cluster_df['levels'].apply(
            lambda x: x[2].strip() if isinstance(x, list) and len(x) > 2 else '(Direct)')

        # 4. Dynamic Path Setup
        path_hierarchy = ['Exam', 'Deck_L1', 'Deck_L2', 'Deck_L3', 'knowledge_state']

        # 5. Group and Aggregate
        cluster_stats = cluster_df.groupby(path_hierarchy).agg(
            card_count=('id', 'count'),
            avg_difficulty=('d', 'mean')
        ).reset_index()

        # 6. Create the multi-layer Sunburst
        fig_cluster = px.sunburst(
            cluster_stats,
            path=path_hierarchy,
            values='card_count',
            color='avg_difficulty',
            color_continuous_scale='RdYlGn_r',
            range_color=[1, 10],
            hover_data={'card_count': True, 'avg_difficulty': ':.2f'}
        )

        fig_cluster.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=750)
        st.plotly_chart(fig_cluster, use_container_width=True)


with tab8:
    # ==========================================
    # 🧠 ADVANCED ANKI ACTION CENTER
    # ==========================================
    st.divider()
    st.header("🧠 Strategic Action Center")
    st.markdown(
        "Bridge your dashboard's machine learning insights directly into Anki. Review the strategy guide below, then generate your custom Filtered Deck payload.")

    # --- 1. STRATEGY DICTIONARY (Metadata & Explanations) ---
    STRATEGIES = {
        "The Quarantine (High Friction + Leeches)":       {
            "target":       "Isolate concepts that cause structural cognitive friction (Difficulty >= 8 & failed multiple times).",
            "instructions": "1. Create this deck at the **end of your study day**.\n2. Do not just spam 'Hard' or 'Again'.\n3. If you fail a card here, edit the note—rewrite it, add context, or break it down."
        },
        "Fragile Knowledge (Low Stability, Due Soon)":    {
            "target":       "Intercept memory decay. Targets cards due in the next 5 days with a memory stability (S) of less than a week.",
            "instructions": "1. Use this as a **warm-up deck** before your main reviews.\n2. By hitting these fragile memories early, you prevent them from lapsing and resetting your progress."
        },
        "Exam Cram (Overdue + Hard)":                     {
            "target":       "High-yield triage. Filters out the easy stuff and forces you to confront the hardest overdue material first.",
            "instructions": "1. Use **only when overwhelmed** by your Anki due count.\n2. Clear this filtered deck first, then delete the deck and let Anki feed you the easy backlog normally."
        },
        "Uncover Blind Spots (Tag-Based)":                {
            "target":       "Force initiation of your most neglected manual categories by targeting tags with the highest volume of 'Unseen' cards.",
            "instructions": "1. Use this when you have dedicated time to **learn new material**.\n2. Turn off regular 'New Cards' in your Anki settings, and exclusively use this to chip away at knowledge gaps."
        },
        "The Neural Oracle (Cursed Concepts)":            {
            "target":       "Preemptively tackle upcoming cards with historical failure rates > 15%, bypassing the standard FSRS schedule.",
            "instructions": "1. Run this **2-3 days before your exam**.\n2. These cards are highly toxic to your retention. Review them out of cycle to ensure they are top-of-mind."
        },
        "Semantic Dark Matter (AI Unseen Cluster)":       {
            "target":       "Attack the densest HDBSCAN semantic cluster of entirely unstudied material. Bypasses manual tags to group cards by text similarity.",
            "instructions": "1. Dive deep into a single, highly related topic you haven't touched yet.\n2. **Important:** In Anki's Filtered Deck settings, select 'Reschedule cards based on my answers'."
        },
        "The Volcano (AI Hardest Cluster)":               {
            "target":       "Deep-dive into the semantic island with the highest average cognitive friction (FSRS Difficulty).",
            "instructions": "1. Use this when you are heavily caffeinated and ready for pain.\n2. This pulls the hardest *topic*, not just random hard cards, allowing you to build contextual understanding."
        },
        "The Reconnaissance Survey (Distinct New Cards)": {
            "target":       "Sample random, unexplored cards across completely different semantic topics to prevent cognitive interference.",
            "instructions": "1. Safely survey the landscape of your blind spots.\n2. Pulls exactly one card from distinct islands (ranked by topic difficulty), forcing your brain to interleave disparate concepts."
        }
    }

    # --- 2. DISPLAY STRATEGY GUIDE UPFRONT ---
    st.markdown("### 📖 Strategy Guide")
    guide_col1, guide_col2 = st.columns(2)

    # Dynamically render the dictionary into a 2-column grid
    strat_keys = list(STRATEGIES.keys())
    for i, strat_name in enumerate(strat_keys):
        col = guide_col1 if i % 2 == 0 else guide_col2
        with col:
            st.info(
                f"**{strat_name}**\n\n**🎯 Target:** {STRATEGIES[strat_name]['target']}\n\n**⚙️ Execution:**\n{STRATEGIES[strat_name]['instructions']}")

    st.divider()

    # --- 3. GENERATOR UI ---
    st.markdown("### 🛠️ Query Generator")
    col_sel, col_param = st.columns([2, 1])

    with col_sel:
        study_strategy = st.selectbox("Select Cognitive Strategy:", strat_keys)

    with col_param:
        cid_limit = st.slider(
            "Max Cards to Pull",
            min_value=10, max_value=150, value=50, step=10,
            help="Anki's search bar can freeze if you pass it thousands of card IDs. Keep this under 150 for smooth performance."
        )

    # --- 4. LOGIC & STRING GENERATION ---
    deck_str = f'"deck:{selected_deck}" ' if selected_deck != "All Decks" else ""
    filter_string = ""

    if study_strategy == "The Quarantine (High Friction + Leeches)":
        filter_string = f"{deck_str}is:review prop:d>=8 prop:lapses>=2"

    elif study_strategy == "Fragile Knowledge (Low Stability, Due Soon)":
        filter_string = f"{deck_str}is:review prop:s<7 prop:due<=5"

    elif study_strategy == "Exam Cram (Overdue + Hard)":
        filter_string = f"{deck_str}is:due prop:due<0 prop:d>=7"

    elif study_strategy == "Uncover Blind Spots (Tag-Based)":
        explore_df = filtered_cards.dropna(subset=['tags']).copy()
        if not explore_df.empty:
            explore_df['tag_list'] = explore_df['tags'].astype(str).str.strip().str.split(r'[ _]')
            exploded = explore_df.explode('tag_list')


            def is_useful_local(t):
                t = str(t).lower().strip()
                return len(t) >= 3 and t not in GLOBAL_STOP_WORDS and not (
                            any(c.isdigit() for c in t) and any(c.isalpha() for c in t) and len(t) > 5)


            exploded = exploded[exploded['tag_list'].apply(is_useful_local)]
            if not exploded.empty:
                stats = exploded.groupby('tag_list').agg(
                    total_cards=('id', 'count'),
                    unseen=('knowledge_state', lambda x: (x == 'Unseen').sum())
                ).reset_index()

                blind_spots = stats[(stats['total_cards'] >= 5) & (stats['unseen'] > 0)].copy()
                if not blind_spots.empty:
                    blind_spots['priority'] = blind_spots['unseen'] * (
                                blind_spots['unseen'] / blind_spots['total_cards'])
                    top_tags = blind_spots.sort_values(by='priority', ascending=False).head(4)['tag_list'].tolist()
                    tag_cluster = " OR ".join([f"tag:*{t}*" for t in top_tags])
                    filter_string = f"{deck_str}is:new ({tag_cluster})"
                else:
                    filter_string = "No tag-based blind spots detected."
            else:
                filter_string = "No usable tags found."

    elif study_strategy == "The Neural Oracle (Cursed Concepts)":
        today = get_local_today()
        future_3_days = today + timedelta(days=3)
        cursed_df = filtered_cards[
            (filtered_cards['due_date'].notna()) & (filtered_cards['due_date'] <= future_3_days) & (
                        filtered_cards['reps'] > 3)].copy()

        if not cursed_df.empty:
            cursed_df['failure_rate'] = cursed_df['lapses'] / cursed_df['reps']
            true_cursed = cursed_df[cursed_df['failure_rate'] > 0.15].sort_values(by='failure_rate', ascending=False)
            if not true_cursed.empty:
                cids = true_cursed['id'].head(cid_limit).astype(str).tolist()
                filter_string = f"{deck_str}is:due (cid:" + " OR cid:".join(cids) + ")"
            else:
                filter_string = "No cursed concepts due soon! You are safe."
        else:
            filter_string = "Not enough review history to calculate cursed concepts."

    elif study_strategy == "Semantic Dark Matter (AI Unseen Cluster)":
        if 'map_df' in globals() and map_df is not None and not map_df.empty and 'cluster' in map_df.columns:
            valid_clusters = map_df[map_df['cluster'] != -1].copy()
            if not valid_clusters.empty:
                cluster_stats = valid_clusters.groupby('cluster').agg(
                    unseen_count=('knowledge_state', lambda x: (x == 'Unseen').sum())
                ).reset_index()

                dark_cluster = cluster_stats.sort_values(by='unseen_count', ascending=False).iloc[0]
                if dark_cluster['unseen_count'] > 0:
                    dark_cards = valid_clusters[(valid_clusters['cluster'] == dark_cluster['cluster']) & (
                                valid_clusters['knowledge_state'] == 'Unseen')]
                    cids = dark_cards['id'].head(cid_limit).astype(str).tolist()
                    filter_string = f"{deck_str}is:new (cid:" + " OR cid:".join(cids) + ")"
                else:
                    filter_string = "No Unseen cards left in any formed semantic clusters!"
            else:
                filter_string = "No semantic clusters formed. View Tab 6 first."
        else:
            filter_string = "⚠️ Map data not found. Please view Tab 6 (3D Maps) to generate the semantic clusters first."

    elif study_strategy == "The Volcano (AI Hardest Cluster)":
        if 'map_df' in globals() and map_df is not None and not map_df.empty and 'cluster' in map_df.columns:
            valid_clusters = map_df[map_df['cluster'] != -1].copy()
            if not valid_clusters.empty:
                cluster_stats = valid_clusters[valid_clusters['d'] > 0].groupby('cluster').agg(
                    avg_difficulty=('d', 'mean')).reset_index()
                if not cluster_stats.empty:
                    hardest_cluster_id = cluster_stats.sort_values(by='avg_difficulty', ascending=False).iloc[0][
                        'cluster']
                    today_str = str(get_local_today())
                    volcano_cards = valid_clusters[
                        (valid_clusters['cluster'] == hardest_cluster_id) &
                        ((valid_clusters['due_date'].astype(str) <= today_str) | (
                                    valid_clusters['knowledge_state'] == 'Unseen'))
                        ]
                    if not volcano_cards.empty:
                        cids = volcano_cards['id'].head(cid_limit).astype(str).tolist()
                        filter_string = f"{deck_str}(cid:" + " OR cid:".join(cids) + ")"
                    else:
                        filter_string = "No due/new cards in your hardest cluster."
                else:
                    filter_string = "Not enough difficulty data."
            else:
                filter_string = "No semantic clusters formed."
        else:
            filter_string = "⚠️ Map data not found. Please view Tab 6 (3D Maps) to generate the semantic clusters first."

    elif study_strategy == "The Reconnaissance Survey (Distinct New Cards)":
        if 'map_df' in globals() and map_df is not None and not map_df.empty and 'cluster' in map_df.columns:
            valid_clusters = map_df[map_df['cluster'] != -1].copy()
            unseen_cards = valid_clusters[valid_clusters['knowledge_state'] == 'Unseen']

            if not unseen_cards.empty:
                # Get avg difficulty for each cluster to prioritize hardest topics
                cluster_diffs = valid_clusters[valid_clusters['d'] > 0].groupby('cluster')['d'].mean().to_dict()

                # Group unseen cards by cluster and shuffle them
                unseen_by_cluster = unseen_cards.groupby('cluster')['id'].apply(
                    lambda x: list(x.sample(frac=1))).to_dict()

                # Sort clusters by difficulty (hardest distinct topics first)
                sorted_clusters = sorted(unseen_by_cluster.keys(), key=lambda k: cluster_diffs.get(k, 0), reverse=True)

                cids = []
                # Round-Robin: take 1 card from each distinct cluster, looping until limit is hit
                while len(cids) < cid_limit and any(unseen_by_cluster.values()):
                    for cluster in sorted_clusters:
                        if len(cids) >= cid_limit:
                            break
                        if unseen_by_cluster[cluster]:
                            cids.append(str(unseen_by_cluster[cluster].pop(0)))

                if cids:
                    filter_string = f"{deck_str}is:new (cid:" + " OR cid:".join(cids) + ")"
                else:
                    filter_string = "Failed to sample new cards."
            else:
                filter_string = "No Unseen cards left in any semantic clusters! Great job."
        else:
            filter_string = "⚠️ Map data not found. Please view Tab 6 (3D Maps) to generate the semantic clusters first."

    # --- 5. RENDER THE PAYLOAD ---
    st.markdown("### 📋 Generated Anki Payload")
    if filter_string.startswith("⚠️") or filter_string.startswith("No") or filter_string.startswith(
            "Not") or filter_string.startswith("Failed"):
        st.warning(filter_string)
    else:
        st.code(filter_string, language="text")
        st.caption(
            "Click the copy icon in the top right of the code block, then paste into Anki's Filtered Deck search bar.")