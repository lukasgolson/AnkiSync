import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
import json
from datetime import date, datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Comprehensive Exam Dashboard", layout="wide", page_icon="🧠")
st.title("🧠 Anki FSRS Exam Prep Dashboard")


# ==========================================
# DATABASE CONNECTION & DATA FETCHING
# ==========================================
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_connection()


@st.cache_data(ttl=3600)
def load_data():
    decks = pd.DataFrame(supabase.table('decks').select('*').execute().data)
    notes = pd.DataFrame(supabase.table('notes').select('id, sfld, tags').execute().data)
    cards = pd.DataFrame(supabase.table('cards').select('*').execute().data)
    revlog = pd.DataFrame(supabase.table('revlog').select('*').execute().data)

    if not revlog.empty:
        revlog['review_datetime'] = pd.to_datetime(revlog['id'], unit='ms')
        revlog['review_date'] = revlog['review_datetime'].dt.date
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

    if not cards.empty:
        fsrs_df = cards['fsrs_data'].apply(parse_fsrs).apply(pd.Series)
        cards = pd.concat([cards, fsrs_df], axis=1)
        cards['s'] = pd.to_numeric(cards['s'], errors='coerce')
        cards['d'] = pd.to_numeric(cards['d'], errors='coerce')

        cards['creation_date'] = pd.to_datetime(cards['id'], unit='ms').dt.date

        cards = cards.merge(decks.rename(columns={'id': 'did', 'name': 'deck_name'}), on='did', how='left')
        cards = cards.merge(notes.rename(columns={'id': 'nid', 'sfld': 'card_front', 'tags': 'tags'}), on='nid',
                            how='left')

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
            cards['forgetting_date'] = cards['last_review_datetime'] + pd.to_timedelta(cards['s'], unit='D')
            cards['forgetting_date'] = cards['forgetting_date'].dt.date

    return decks, notes, cards, revlog


with st.spinner("Loading Anki data from Supabase..."):
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🧠 FSRS Decay", "🔍 Problem Cards", "🏷️ Tag Analytics", "🌌 3D Knowledge Map"
])

# --- TAB 1: OVERVIEW & PROGRESS ---
with tab1:
    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        st.subheader("Daily Review Volume & Cumulative Total")
        if not filtered_revlog.empty:
            # 1. Prepare data for stacked bars
            daily_reviews = filtered_revlog.groupby(['review_date', 'ease_label']).size().reset_index(name='count')

            # 2. Prepare data for cumulative line
            daily_totals = filtered_revlog.groupby('review_date').size().reset_index(name='daily_total').sort_values(
                'review_date')
            daily_totals['cumulative'] = daily_totals['daily_total'].cumsum()

            # 3. Create figure with secondary Y-axis
            fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

            color_map = {'Again': '#ef4444', 'Hard': '#f59e0b', 'Good': '#22c55e', 'Easy': '#3b82f6'}

            # Add stacked bars
            for ease in ['Again', 'Hard', 'Good', 'Easy']:
                ease_data = daily_reviews[daily_reviews['ease_label'] == ease]
                if not ease_data.empty:
                    fig_bar.add_trace(
                        go.Bar(x=ease_data['review_date'], y=ease_data['count'], name=ease,
                               marker_color=color_map[ease]),
                        secondary_y=False,
                    )

            # Add cumulative line
            fig_bar.add_trace(
                go.Scatter(
                    x=daily_totals['review_date'], y=daily_totals['cumulative'],
                    name='Cumulative Reviews', mode='lines',
                    line=dict(color='#8b5cf6', width=3)  # Distinct purple line
                ),
                secondary_y=True,
            )

            # Format layout
            fig_bar.update_layout(barmode='stack', hovermode="x unified", margin=dict(t=10))
            fig_bar.update_yaxes(title_text="Daily Reviews", secondary_y=False)
            fig_bar.update_yaxes(title_text="Total Cumulative", secondary_y=True, showgrid=False)

            st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("Knowledge Mastery Breakdown")
        state_counts = filtered_cards['knowledge_state'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        fig_pie = px.pie(
            state_counts, values='Count', names='State', hole=0.4,
            color='State', color_discrete_map=state_colors
        )
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

# --- TAB 2: FSRS & MEMORY DECAY ---
with tab2:
    st.subheader("Predicted Memory Decay Timeline")
    today = date.today()
    future_forgets = filtered_cards[filtered_cards['forgetting_date'] >= today]
    if not future_forgets.empty:
        decay_counts = future_forgets.groupby('forgetting_date').size().reset_index(name='cards_decaying')
        fig_decay = px.area(decay_counts, x='forgetting_date', y='cards_decaying', color_discrete_sequence=['#ef4444'])
        fig_decay.update_xaxes(range=[today, today + pd.Timedelta(days=180)])
        st.plotly_chart(fig_decay, use_container_width=True)

    st.divider()
    st.subheader("FSRS Memory State: Difficulty vs. Stability")
    fsrs_plot_df = filtered_cards.dropna(subset=['d', 's', 'card_front'])
    if not fsrs_plot_df.empty:
        fig_scatter = px.scatter(
            fsrs_plot_df, x='d', y='s', hover_data=['card_front', 'lapses', 'reps'],
            color='deck_name' if selected_deck == "All Decks" else None, opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: PROBLEM CARDS ---
with tab3:
    st.subheader("Leech & High-Difficulty Cards")
    fsrs_plot_df = filtered_cards.dropna(subset=['d', 's', 'card_front'])
    if not fsrs_plot_df.empty:
        problem_cards = fsrs_plot_df[
            ['card_front', 'deck_name', 'knowledge_state', 'd', 's', 'lapses', 'forgetting_date']].sort_values(
            by=['d', 'lapses'], ascending=[False, False])
        problem_cards = problem_cards.rename(columns={'d': 'Difficulty', 's': 'Stability (Days)'})
        st.dataframe(problem_cards, use_container_width=True, hide_index=True, height=500)

# --- TAB 4: TAG ANALYTICS ---
with tab4:
    st.subheader("🏷️ Subject Difficulty by Tag")
    tag_df = filtered_cards.dropna(subset=['d', 'tags']).copy()
    if not tag_df.empty:
        tag_df['tag_list'] = tag_df['tags'].astype(str).str.strip().str.split(' ')
        exploded_tags = tag_df.explode('tag_list')
        exploded_tags = exploded_tags[exploded_tags['tag_list'] != '']
        if not exploded_tags.empty:
            tag_stats = exploded_tags.groupby('tag_list').agg(avg_difficulty=('d', 'mean'), avg_stability=('s', 'mean'),
                                                              card_count=('id', 'count')).reset_index()
            tag_stats = tag_stats[tag_stats['card_count'] >= 5]
            hardest_tags = tag_stats.sort_values(by='avg_difficulty', ascending=True).tail(20)
            if not hardest_tags.empty:
                col_tag_chart, col_tag_data = st.columns([3, 2])
                with col_tag_chart:
                    fig_tags = px.bar(hardest_tags, x='avg_difficulty', y='tag_list', orientation='h',
                                      color='avg_difficulty', color_continuous_scale='Reds')
                    fig_tags.update_xaxes(range=[1, 10])
                    st.plotly_chart(fig_tags, use_container_width=True)
                with col_tag_data:
                    st.dataframe(tag_stats.sort_values(by='avg_difficulty', ascending=False), use_container_width=True,
                                 hide_index=True, height=500)

# --- TAB 5: 3D KNOWLEDGE MAP ---
with tab5:
    st.subheader("🌌 3D Semantic Knowledge Map")
    st.markdown("""
    **How to read this map:**
    * **The Geometry (X, Y, Z Axes):** We used Principal Component Analysis (PCA) to combine your flashcard text and tags, compressing thousands of concepts into 3 main topics of variance. Cards sitting close to each other conceptually mean the same thing.
    * **The Color (Difficulty):** Dark Blue/Green means the card is easy. **Bright Red/Orange means the card is brutally hard (Difficulty > 8).** *Spin the graph to look for "Red Galaxies"—these are entire conceptual topics you are struggling with!*
    """)

    map_df = filtered_cards.dropna(subset=['card_front', 'd']).copy()

    if len(map_df) > 10:
        with st.spinner("Calculating 3D semantic vectors..."):
            map_df['combined_features'] = map_df['card_front'].astype(str) + " " + map_df['tags'].astype(str).fillna("")
            vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
            X = vectorizer.fit_transform(map_df['combined_features'])

            svd = TruncatedSVD(n_components=3, random_state=42)
            coords = svd.fit_transform(X)

            map_df['PC1 (Primary Topic Variance)'] = coords[:, 0]
            map_df['PC2 (Secondary Topic Variance)'] = coords[:, 1]
            map_df['PC3 (Tertiary Topic Variance)'] = coords[:, 2]

            fig_map = px.scatter_3d(
                map_df,
                x='PC1 (Primary Topic Variance)',
                y='PC2 (Secondary Topic Variance)',
                z='PC3 (Tertiary Topic Variance)',
                color='d',
                color_continuous_scale='Turbo',
                hover_name='deck_name',
                hover_data={
                    'PC1 (Primary Topic Variance)':   False,
                    'PC2 (Secondary Topic Variance)': False,
                    'PC3 (Tertiary Topic Variance)':  False,
                    'card_front':                     True,
                    'tags':                           True,
                    'd':                              True,
                    's':                              True,
                    'knowledge_state':                True
                },
                opacity=0.7,
                height=800
            )

            fig_map.update_traces(marker=dict(size=4))
            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("You need at least 10 reviewed cards to generate a 3D semantic map.")