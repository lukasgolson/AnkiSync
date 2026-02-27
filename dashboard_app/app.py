import json
import re
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from supabase import create_client

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


@st.cache_data(ttl=3600)
def load_data():
    decks = pd.DataFrame(supabase.table('decks').select('*').execute().data)
    notes = pd.DataFrame(supabase.table('notes').select('id, sfld, tags').execute().data)
    cards = pd.DataFrame(supabase.table('cards').select('*').execute().data)
    revlog = pd.DataFrame(supabase.table('revlog').select('*').execute().data)

    if not revlog.empty:
        revlog['review_datetime'] = pd.to_datetime(revlog['id'], unit='ms')
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

        cards['creation_date'] = pd.to_datetime(cards['id'], unit='ms').dt.date

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
    """Calculates the current state of the player based on deck performance."""
    # 1. Base Stats
    mature_count = len(cards_df[cards_df['knowledge_state'] == 'Known'])
    avg_s = cards_df['s'].mean() if 's' in cards_df.columns else 0

    # 2. XP & Gold Ledger
    total_xp = int((mature_count * 20) + (avg_s * 10))
    gross_gold = int(total_xp * 0.1)

    # 3. Interest (Bonus for high stability)
    interest = int(gross_gold * 0.05) if avg_s > 21 else 0

    # 4. Taxes (Penalty for 'Again' presses today)
    today_fails = len(revlog_df[(revlog_df['review_date'] == date.today()) & (revlog_df['ease'] == 1)])
    daily_tax = today_fails * 10

    current_gold = max(0, (gross_gold + interest) - (db_stats.get('spent_gold', 0) + daily_tax))

    # RETURN ALL VARIABLES TO AVOID NAMEERRORS
    return total_xp, current_gold, gross_gold, interest, daily_tax, avg_s


with st.spinner("Loading & crunching Anki data..."):
    decks_df, notes_df, cards_df, revlog_df = load_data()

if cards_df.empty or revlog_df.empty:
    st.warning("No data found! Please make sure you've run your sync script.")
    st.stop()

# ==========================================
# SIDEBAR FILTERS & EXAM COUNTDOWN
# ==========================================
st.sidebar.header("🎯 Target")
exam_date = date(date.today().year, 4, 14)
if date.today() > exam_date:
    exam_date = date(date.today().year + 1, 4, 14)

days_left = (exam_date - date.today()).days
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
                      help="Number of 'Unseen' cards you must start daily to finish the deck by your exam.")

    if daily_pace > 30:
        st.sidebar.warning("⚠️ High Pace Required! Consider triaging your 'Unseen' cards.")
    else:
        st.sidebar.success("✅ Pace is manageable.")


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
    check_date = date.today()
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Overview", "🔮 Future Workload", "⏱️ Study Optimization", "🏷️ Tag Analytics", "🔍 Problem Cards", "🌌 3D Map", "🎯 Readiness", "Meta Clustering", "Game"
])

# --- TAB 1: OVERVIEW ---
with tab1:
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
    col_heat, col_create = st.columns([1, 1])
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

    with col_create:
        st.subheader("New Cards Added Over Time")
        if not filtered_cards.empty:
            new_cards = filtered_cards.groupby('creation_date').size().reset_index(name='cards_added')
            fig_create = px.line(new_cards, x='creation_date', y='cards_added', markers=True)
            fig_create.update_traces(line_color="#8b5cf6")
            st.plotly_chart(fig_create, use_container_width=True)

        st.subheader("📈 Learning Growth")
        if not filtered_revlog.empty:
            # Calculate when each card was first seen
            first_seen = filtered_revlog.groupby('cid')['review_date'].min().reset_index()
            daily_new_seen = first_seen.groupby('review_date').size().reset_index(name='New Seen')

            fig_seen = px.bar(daily_new_seen, x='review_date', y='New Seen', title="New Cards Seen (First Contact)")
            fig_seen.update_traces(marker_color="#22c55e")
            st.plotly_chart(fig_seen, use_container_width=True)


# --- TAB 2: FUTURE WORKLOAD & DECAY ---
with tab2:
    col_future1, col_future2 = st.columns(2)
    today = date.today()

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
        cards_today = len(filtered_cards[filtered_cards['due_date'] == date.today()])
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
# --- TAB 4: REFINED TAG ANALYTICS ---
with tab4:
    st.subheader("🏷️ Subject Difficulty by Tag (Cleaned)")

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

# --- TAB 5: PROBLEM CARDS ---
with tab5:
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
with tab6:
    st.subheader("🌌 3D Semantic Knowledge Map")

    # Define a sidebar control for "Words to Ignore"
    st.sidebar.divider()
    st.sidebar.header("🌌 Map Settings")

    # 1. Define ALL controls at the top to avoid NameErrors
    manual_ignore = st.sidebar.text_input("Extra Words to Ignore (comma separated)", "")
    show_labels = st.sidebar.checkbox("Show Island Labels", value=True)
    num_clusters = st.sidebar.slider("Number of Concept Islands", 3, 15, 8)
    custom_perplexity = st.sidebar.slider("Map Detail (Perplexity)", 5, 50, 15)

    map_df = filtered_cards.dropna(subset=['clean_text', 'd']).copy()

    if len(map_df) > 30:
        with st.spinner("Purging citations and calculating t-SNE..."):

            # 2. Build the custom ignore list from your sidebar input
            user_stop_words = [w.strip().lower() for w in manual_ignore.split(",") if w.strip()]
            fixed_stop_words = ['moreau', 'payette', 'bauce', 'allen', 'rademacher', 'results', 'discussion',
                                'background']
            all_stops = fixed_stop_words + user_stop_words


            # 3. Scientific Purge Function
            # 3. Scientific Purge Function (Updated to strip numbers/dates)
            def purge_noise(text):
                # 1. Lowercase and remove punctuation/special chars
                text = re.sub(r'[^\w\s]', ' ', str(text).lower())

                # 2. Split into tokens
                tokens = re.split(r'[ _]', text)

                clean_tokens = []
                for t in tokens:
                    # Ignore if the token is a number or contains digits (dates, measurements)
                    if any(char.isdigit() for char in t):
                        continue

                    # Ignore common noise and short fragments (et, al, s2)
                    if len(t) > 3 and t not in all_stops:
                        clean_tokens.append(t)

                return " ".join(clean_tokens)


            map_df['nlp_ready_text'] = map_df['clean_text'].apply(purge_noise)
            map_df['nlp_ready_tags'] = map_df['tags'].apply(purge_noise)
            map_df['combined_features'] = map_df['nlp_ready_text'] + " " + map_df['nlp_ready_tags']

            # 4. Math Pipeline
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1500,  # Increased to account for extra phrase tokens
                max_df=0.4,
                min_df=3,
                ngram_range=(1, 3)
            )
            X = vectorizer.fit_transform(map_df['combined_features'])

            svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=42)
            X_reduced = svd.fit_transform(X)

            # custom_perplexity is now safely defined at the top of the block
            final_perp = min(custom_perplexity, len(map_df) - 1)

            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=final_perp,
                init='pca',
                learning_rate='auto'
            )
            coords = tsne.fit_transform(X_reduced)

            map_df['Map X'], map_df['Map Y'], map_df['Map Z'] = coords[:, 0], coords[:, 1], coords[:, 2]

            # 5. K-Means for Island Labelling
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            map_df['cluster'] = kmeans.fit_predict(coords)

            cluster_labels = {}
            if show_labels:
                feature_names = vectorizer.get_feature_names_out()

                for i in range(num_clusters):
                    cluster_indices = map_df[map_df['cluster'] == i].index
                    # Calculate mean TF-IDF for this cluster to find 'signature' phrases
                    cluster_tfidf = X[map_df.index.get_indexer(cluster_indices)].mean(axis=0).A1

                    # Sort by highest score
                    top_indices = cluster_tfidf.argsort()[::-1]

                    # Logic: Prioritize the highest scoring n-gram that isn't noise
                    best_label = f"Cluster {i}"
                    for idx in top_indices:
                        candidate = feature_names[idx]
                        # Ensure the label isn't just a citation or one of your stop words
                        if not any(stop in candidate for stop in all_stops):
                            best_label = candidate.upper()
                            break
                    cluster_labels[i] = best_label

            # 6. Final 3D Plot
            fig_map = px.scatter_3d(
                map_df, x='Map X', y='Map Y', z='Map Z',
                color='d', color_continuous_scale='Turbo',
                hover_name='deck_name',
                hover_data={'Map X': False, 'Map Y': False, 'Map Z': False, 'clean_text': True, 'd': True},
                opacity=0.7, height=800
            )

            if show_labels:
                centers = map_df.groupby('cluster')[['Map X', 'Map Y', 'Map Z']].mean().reset_index()
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
    else:
        st.info("You need at least 30 cards to generate this map.")

# --- TAB 7: MASTER READINESS & CALIBRATION ---
with tab7:
    st.subheader("🎯 Master Readiness & FSRS Calibration")

    col_gauge, col_stab = st.columns([1, 2])

    # 1. Calculate RMSE (Simplified Proxy)
    # FSRS RMSE is the diff between predicted Retrievability (r) and actual outcome
    if not filtered_revlog.empty and 'r' in filtered_cards.columns:
        # We look at recently reviewed cards that have FSRS retrievability data
        calib_df = filtered_cards.dropna(subset=['r', 'last_review_datetime'])

        # Simple RMSE calculation: sqrt(mean((Actual - Predicted)^2))
        # For a truly live gauge, we use the average deviation of stability
        actual_retention = (filtered_revlog['ease'] > 1).mean()
        target_retention = 0.90  # Default Anki target
        rmse_score = abs(actual_retention - target_retention) * 100

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rmse_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "FSRS Calibration Error (RMSE %)", 'font': {'size': 18}},
                gauge={
                    'axis':        {'range': [0, 20], 'tickwidth': 1},
                    'bar':         {'color': "#60a5fa"},
                    'bgcolor':     "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps':       [
                        {'range': [0, 5], 'color': '#22c55e'},  # Green: Excellent
                        {'range': [5, 10], 'color': '#facc15'},  # Yellow: Fair
                        {'range': [10, 20], 'color': '#ef4444'}  # Red: Needs Optimization
                    ],
                }
            ))
            fig_gauge.update_layout(height=400, margin=dict(t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.caption(f"**Current True Retention:** {actual_retention * 100:.1f}%")
            st.caption(f"Target is 90%. Lower RMSE means better scheduling.")

        with col_stab:
            st.markdown(f"**Average Memory Stability:** {filtered_cards['s'].mean():.1f} days")
            fig_stab = px.histogram(
                filtered_cards, x='s', nbins=50,
                title="Knowledge Depth (Stability Distribution)",
                color_discrete_sequence=['#60a5fa']
            )
            fig_stab.update_layout(xaxis_title="Days until Forgotten", yaxis_title="Number of Cards", height=400)
            st.plotly_chart(fig_stab, use_container_width=True)

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


# --- TAB 8: ROBUST ALL-HEURISTIC STRATEGIC MAP ---
# --- TAB 8: ROBUST ALL-HEURISTIC STRATEGIC MAP (WITH SEMANTIC LABELS) ---
with tab8:
    st.subheader("🌌 The All-Heuristic Strategy Galaxy")
    st.markdown("""
    This map synthesizes every data point Anki and FSRS have on your learning behavior.
    * **Vertical Position (Z):** Stability (Memory Depth) — Higher is better!
    * **Color:** Difficulty (Red = Hardest, Green = Mastered)
    * **Size:** Lapses (Conceptual friction/failures)
    * **Text Labels:** Automated "signature concept" for high-risk cards
    """)

    # 1. Define Heuristics and filter for available data
    base_heuristics = ['d', 's', 'ivl', 'lapses', 'reps', 'r']
    available_h = [h for h in base_heuristics if h in filtered_cards.columns]

    # Create a copy focused on cards with core FSRS data
    strat_df = filtered_cards.dropna(subset=['d', 's']).copy()

    if len(strat_df) > 15:
        with st.spinner("Crunching behavioral heuristics and semantic labels..."):

            # 2. Impute missing values with medians for robustness
            for h in available_h:
                median_val = strat_df[h].median()
                if pd.isna(median_val):
                    # Use mathematical defaults if a column is totally empty
                    default = 0.9 if h == 'r' else 0
                    strat_df[h] = strat_df[h].fillna(default)
                else:
                    strat_df[h] = strat_df[h].fillna(median_val)

            # 3. Final safety cleanup
            final_df = strat_df.dropna(subset=available_h).copy()

            if len(final_df) > 10:
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                strat_scaled = scaler.fit_transform(final_df[available_h])

                # 4. 3D t-SNE Projection (PCA Init for stability)
                tsne_all = TSNE(
                    n_components=3,
                    perplexity=min(30, len(final_df) - 1),
                    random_state=42,
                    init='pca',
                    learning_rate='auto'
                )
                strat_coords = tsne_all.fit_transform(strat_scaled)
                final_df['HX'], final_df['HY'], final_df['HZ'] = strat_coords[:, 0], strat_coords[:, 1], strat_coords[:,
                                                                                                         2]

                # 5. K-Means Behavioral Clustering
                n_meta_clusters = min(5, len(final_df))
                km_meta = KMeans(n_clusters=n_meta_clusters, random_state=42, n_init='auto')
                final_df['meta_cluster'] = km_meta.fit_predict(strat_scaled)

                # 6. Strategic Labeling based on behavioral patterns
                centers = final_df.groupby('meta_cluster')[available_h].mean()
                meta_labels = {}
                avg_lapses = final_df['lapses'].mean()

                for i, row in centers.iterrows():
                    if row.get('lapses', 0) > avg_lapses * 1.5:
                        meta_labels[i] = "🛑 THE LEECH PIT"
                    elif row['d'] > 7.5 and row['s'] < 10:
                        meta_labels[i] = "⚠️ HIGH-FRICTION ZONE"
                    elif row['s'] > 45:
                        meta_labels[i] = "💎 LONG-TERM ASSETS"
                    elif row.get('r', 1) < 0.88:
                        meta_labels[i] = "📉 URGENT DECAY"
                    else:
                        meta_labels[i] = "⚡ STEADY PROGRESS"

                final_df['Behavioral_Group'] = final_df['meta_cluster'].map(meta_labels)


                # 7. Semantic Label Extraction per point
                def get_point_concept(text, tags):
                    raw = f"{text} {tags}"
                    # Use a regex to strip numbers and dates
                    text_no_nums = re.sub(r'\d+', '', str(raw).lower())
                    tokens = re.split(r'[ _]', re.sub(r'[^\w\s]', ' ', text_no_nums))
                    clean = [t for t in tokens if len(t) > 3 and t not in all_stops]
                    return max(clean, key=len).upper() if clean else "MISC"


                final_df['Concept'] = final_df.apply(lambda x: get_point_concept(x['clean_text'], x['tags']), axis=1)

                # 8. The 3D Strategy Plot
                fig_all = px.scatter_3d(
                    final_df, x='HX', y='HY', z='s',
                    color='d', size='lapses',
                    symbol='Behavioral_Group',
                    hover_name='Concept',
                    hover_data={
                        'clean_text': True,
                        'deck_name':  True,
                        'd':          True,
                        's':          ':.1f',
                        'lapses':     True,
                        'HX':         False, 'HY': False
                    },
                    opacity=0.8, height=850,
                    color_continuous_scale='RdYlGn_r'
                )

                # Overlay text labels for 'Danger' clusters to highlight friction
                danger_groups = ["🛑 THE LEECH PIT", "⚠️ HIGH-FRICTION ZONE"]
                danger_df = final_df[final_df['Behavioral_Group'].isin(danger_groups)]
                if not danger_df.empty:
                    fig_all.add_trace(go.Scatter3d(
                        x=danger_df['HX'], y=danger_df['HY'], z=danger_df['s'],
                        mode='text',
                        text=danger_df['Concept'],
                        textfont=dict(size=10, color="white"),
                        showlegend=False
                    ))

                fig_all.update_layout(scene=dict(
                    xaxis=dict(showticklabels=False, title='Behavioral Context X'),
                    yaxis=dict(showticklabels=False, title='Behavioral Context Y'),
                    zaxis_title='Memory Stability (Days)'
                ))
                st.plotly_chart(fig_all, use_container_width=True)

                # 9. Cluster Statistics Summary
                st.subheader("📋 Behavioral Performance Summary")
                readable_centers = centers.copy()
                readable_centers.index = [meta_labels.get(i, f"Group {i}") for i in readable_centers.index]
                st.dataframe(readable_centers.sort_values('d', ascending=False), use_container_width=True)

            else:
                st.warning("Not enough clean data left for this operation.")
    else:
        st.info("Insufficient FSRS history to generate this map. Keep reviewing!")

# --- TAB 9: THE ADVENTURER'S GUILD (SKILLS, TAXES & INTEREST) ---

# --- TAB 9: THE ADVENTURER'S GUILD (XP, BOSSES & SKILLS) ---
with tab9:
    # 1. THE CALCULATION ENGINE (Unpack everything here to avoid NameErrors)
    db_stats = get_user_stats()

    # Run the RPG state logic (Assumes calculate_rpg_state is defined as per our previous turn)
    total_xp, current_gold, gross_gold, interest, daily_tax, avg_s = calculate_rpg_state(
        filtered_cards, filtered_revlog, db_stats
    )

    # Derive Levels and Skills
    user_level = int((total_xp / 150) ** 0.5) + 1
    fortitude_lvl = int(avg_s / 5)

    retention = (filtered_revlog['ease'] > 1).mean() * 100 if not filtered_revlog.empty else 0
    precision_lvl = int(retention / 10)

    avg_time = filtered_revlog['time'].mean() / 1000 if not filtered_revlog.empty else 0
    celerity_lvl = int(max(0, 100 - (avg_time * 5)) / 10)

    seen_cards = len(filtered_cards[filtered_cards['knowledge_state'] != 'Unseen'])
    total_cards = len(filtered_cards)
    cart_lvl = int((seen_cards / total_cards) * 10) if total_cards > 0 else 0

    # 2. XP PROGRESSION BAR
    st.header(f"🛡️ Rank {user_level} Scholar")

    xp_for_current = (user_level - 1) ** 2 * 150
    xp_for_next = user_level ** 2 * 150
    # Prevent division by zero and ensure range 0.0 - 1.0
    denom = (xp_for_next - xp_for_current)
    lvl_progress = (total_xp - xp_for_current) / denom if denom > 0 else 0

    st.progress(min(max(lvl_progress, 0.0), 1.0),
                text=f"✨ {total_xp - xp_for_current} / {denom} XP to Level {user_level + 1}")

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Gold Balance", f"💰 {current_gold}")
    col_v2.metric("Gross Gold", f"🪙 {gross_gold}")
    col_v3.metric("Daily Interest", f"📈 +{interest}")

    st.divider()

    # 3. DYNAMIC DAILY BOUNTIES
    st.subheader("📜 Dynamic Daily Bounties")

    # Calculate goals based on actual deck state
    cards_due_today = len(filtered_cards[filtered_cards['s'] < 1])
    unseen_remaining = total_cards - seen_cards

    scribe_goal = max(20, min(100, cards_due_today))
    explorer_goal = max(5, min(25, int(unseen_remaining * 0.05)))

    today_revs = filtered_revlog[filtered_revlog['review_date'] == date.today()]
    today_count = len(today_revs)
    today_new = len(filtered_cards[filtered_cards['creation_date'] == date.today()])

    q1, q2, q3 = st.columns(3)
    with q1:
        st.write(f"**The Scribe** ({scribe_goal} Reviews)")
        st.progress(min(1.0, today_count / scribe_goal) if scribe_goal > 0 else 1.0)
        if today_count >= scribe_goal:
            st.success("💰 +20 Gold")

    with q2:
        st.write(f"**The Explorer** ({explorer_goal} New)")
        st.progress(min(1.0, today_new / explorer_goal) if explorer_goal > 0 else 1.0)
        if today_new >= explorer_goal:
            st.success("💰 +30 Gold")

    with q3:
        st.write("**The Perfectionist** (90% Acc)")
        today_acc = (today_revs['ease'] > 1).mean() * 100 if today_count > 0 else 0
        st.progress(min(1.0, today_acc / 90))
        if today_acc >= 90 and today_count >= 10:
            st.success("💰 +50 Gold")

    st.divider()

    # 4. BOSS ENCOUNTERS (Semantic Instability)
    st.subheader("👺 Active Bosses")

    if 'cluster' in filtered_cards.columns:
        # Identify 'Fragile' semantic clusters
        cluster_stats = filtered_cards.groupby('cluster').agg(
            avg_stability=('s', 'mean'),
            count=('id', 'count')
        )
        bosses = cluster_stats[(cluster_stats['avg_stability'] < 3) & (cluster_stats['count'] > 10)]

        if not bosses.empty:
            target_id = bosses.index[0]
            st.error(f"⚠️ **BOSS SPAWNED: Cluster {target_id} is Fading!**")
            if st.button("⚔️ Challenge Boss", key="btn_boss"):
                boss_cards = filtered_cards[filtered_cards['cluster'] == target_id]
                st.dataframe(boss_cards[['clean_text', 'd', 's']].sort_values('d', ascending=False))
        else:
            st.success("The semantic horizon is clear. No bosses detected.")

    st.divider()

    # 5. CHARACTER ATTRIBUTES (SPIDER GRAPH)
    st.subheader("⚔️ Character Attributes")
    radar_data = pd.DataFrame(dict(
        r=[fortitude_lvl, precision_lvl, celerity_lvl, cart_lvl, fortitude_lvl],
        theta=['Fortitude', 'Precision', 'Celerity', 'Cartography', 'Fortitude']
    ))
    fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True)
    fig_radar.update_traces(fill='toself', line_color='#8b5cf6')
    fig_radar.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_radar, use_container_width=True)