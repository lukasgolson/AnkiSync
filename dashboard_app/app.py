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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview", "🔮 Future Workload", "⏱️ Study Optimization", "🏷️ Tag Analytics", "🔍 Problem Cards", "🌌 3D Map", "🎯 Readiness"
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
        st.subheader("⏰ Retention by Hour")
        st.markdown("The time of day when my brain is most primed for learning.")
        if not filtered_revlog.empty:
            hourly_stats = filtered_revlog.groupby('hour').agg(
                total_reviews=('id', 'count'),
                passed_reviews=('ease', lambda x: (x > 1).sum())
            ).reset_index()
            hourly_stats['retention'] = (hourly_stats['passed_reviews'] / hourly_stats['total_reviews']) * 100

            # Filter out extreme outliers (e.g., hours where you only did 2 reviews ever)
            hourly_stats = hourly_stats[hourly_stats['total_reviews'] > 10]

            fig_hour = px.line(
                hourly_stats, x='hour', y='retention', markers=True,
                labels={'hour': 'Hour of Day (24h)', 'retention': 'True Retention %'},
                title="When am I most accurate?"
            )
            fig_hour.update_traces(line_color="#10b981", line_width=3)
            # Add a baseline threshold line for 90% FSRS target
            fig_hour.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90% FSRS Target")
            st.plotly_chart(fig_hour, use_container_width=True)

    with col_opt2:
        st.subheader("🎯 Button Bias Analysis")
        st.markdown("My historic button press history")
        if not filtered_revlog.empty:
            button_counts = filtered_revlog['ease_label'].value_counts().reset_index()
            button_counts.columns = ['Button', 'Count']
            fig_buttons = px.bar(
                button_counts, x='Button', y='Count', color='Button',
                color_discrete_map=ease_colors,
                text='Count'
            )
            fig_buttons.update_traces(textposition='outside')
            st.plotly_chart(fig_buttons, use_container_width=True)

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

# --- TAB 6: 3D t-SNE KNOWLEDGE MAP ---
# --- TAB 6: 3D t-SNE KNOWLEDGE MAP (With Auto-Labelling) ---
# --- TAB 6: CLEANED 3D t-SNE MAP ---
# --- TAB 6: CLEANED 3D t-SNE MAP ---
with tab6:
    st.subheader("🌌 3D Semantic Knowledge Map (Scientific Concepts)")

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
            def purge_noise(text):
                tokens = re.split(r'[ _]', str(text).lower())
                clean_tokens = [
                    t for t in tokens
                    if len(t) > 3
                       and not (any(char.isdigit() for char in t) and any(char.isalpha() for char in t))
                       and t not in all_stops
                ]
                return " ".join(clean_tokens)


            map_df['nlp_ready_text'] = map_df['clean_text'].apply(purge_noise)
            map_df['nlp_ready_tags'] = map_df['tags'].apply(purge_noise)
            map_df['combined_features'] = map_df['nlp_ready_text'] + " " + map_df['nlp_ready_tags']

            # 4. Math Pipeline
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, max_df=0.5, min_df=2)
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
                for i in range(num_clusters):
                    cluster_text = " ".join(map_df[map_df['cluster'] == i]['combined_features'])
                    words = [w for w in re.findall(r'\w+', cluster_text.lower()) if len(w) > 4 and w not in all_stops]
                    if words:
                        most_common = max(set(words), key=words.count)
                        cluster_labels[i] = most_common.upper()
                    else:
                        cluster_labels[i] = f"Group {i}"

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

# --- TAB 7: MASTER READINESS SUPER-CLUSTER ---
with tab7:
    st.subheader("🎯 Master Readiness Super-Cluster")
    st.markdown("""
    **The 'One Big Picture' of my Exam Preparedness:**
    This sunburst cluster represents my entire brain's current state regarding the exam. 
    * **Size** = Number of cards in that category. 
    * **Color** = Average FSRS Difficulty (Green = Easy/Mastered, Red = Brutally Hard). 
    * *Interactive:* Click on any slice (like a specific deck) to zoom into that sub-cluster!
    """)

    # Drop rows missing the core categorization data
    cluster_df = filtered_cards.dropna(subset=['deck_name', 'knowledge_state', 'd']).copy()

    if not cluster_df.empty:
        # Group the data hierarchically: Deck -> Knowledge State
        cluster_stats = cluster_df.groupby(['deck_name', 'knowledge_state']).agg(
            card_count=('id', 'count'),
            avg_difficulty=('d', 'mean')
        ).reset_index()

        # Add a root node so everything clusters into one giant entity
        cluster_stats['Exam'] = 'April 14 Exam'

        fig_cluster = px.sunburst(
            cluster_stats,
            path=['Exam', 'deck_name', 'knowledge_state'],
            values='card_count',
            color='avg_difficulty',
            # RdYlGn_r is a Red-Yellow-Green scale, reversed so low difficulty (1) is Green and high (10) is Red
            color_continuous_scale='RdYlGn_r',
            range_color=[1, 10],  # Lock the color scale to the strict 1-10 FSRS difficulty range
            hover_data={'card_count': True, 'avg_difficulty': ':.2f'}
        )

        fig_cluster.update_layout(
            margin=dict(t=20, l=10, r=10, b=10),
            height=750
        )

        # Clean up the hover text for readability
        fig_cluster.update_traces(
            hovertemplate='<b>%{id}</b><br>Cards: %{value}<br>Avg Difficulty: %{color:.2f}'
        )

        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Not enough data to generate the master cluster.")