import os, re
from pathlib import Path
from datetime import datetime
from collections import Counter
from unittest import result
import pandas as pd
import streamlit as st
import plotly.express as px

px.defaults.template = "plotly_dark"
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

load_dotenv()
st.set_page_config(page_title="Pakistani Social Media Content Detective", page_icon="🕵️", layout="wide")


st.markdown("""
<style>
/* Consistent dark dashboard background and readable text */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #0e1117 !important;
    color: #f0f2f6 !important;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #161b22 !important;
}
[data-testid="stSidebar"] * {
    color: #f0f2f6;
}

/* Top navigation tabs: dark background and visible labels */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 12px 16px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] div,
[data-testid="stTabs"] [data-baseweb="tab"] span {
    color: #e6edf3 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #252d38 !important;
    color: #ffffff !important;
    border-bottom: 3px solid #4f8cff !important;
}
[data-testid="stTabs"] [aria-selected="true"] p,
[data-testid="stTabs"] [aria-selected="true"] div,
[data-testid="stTabs"] [aria-selected="true"] span {
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #161b22 !important;
    padding: 18px !important;
    border-radius: 14px !important;
    border: 1px solid #30363d !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.20);
    transition: transform 180ms ease, box-shadow 180ms ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetric"] * {
    color: #f0f2f6 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 15px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
}

/* Headings, captions, and regular content */
h1, h2, h3, h4, p, label, li {
    color: #f0f2f6;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #b1bac4 !important;
}

/* Inputs and select boxes */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #0e1117 !important;
    color: #f0f2f6 !important;
    border-color: #30363d !important;
}
input, textarea {
    color: #f0f2f6 !important;
}

/* Keep alerts and containers readable in dark mode */
[data-testid="stAlert"] {
    border: 1px solid #30363d;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)


st.title("Pakistani Social Media Content Detective")
st.caption("YouTube analytics • Customer sentiment • Clothing content recommendations")
st.info("Research dashboard for Pakistani clothing brands. Public YouTube data is a sample, not a guarantee of future performance.")

# Small labelled starter set: replace/expand with a properly labelled Urdu/Pashto/Roman Urdu dataset for research-grade results.
TRAIN_TEXTS = [
 "beautiful dress amazing quality", "love this suit", "fabric is very good", "excellent design and stitching",
 "nice collection beautiful color", "will buy this dress", "delivery was fast and good", "زبردست لباس بہت اچھا",
 "بہت خوبصورت سوٹ", "quality is excellent", "amazing outfit", "great design",
 "bad quality", "very expensive and poor fabric", "disappointed with this dress", "not worth the price",
 "stitching is terrible", "I hate this product", "بہت خراب کوالٹی", "ناپسند آیا",
 "price is too high", "poor service", "okay product", "average dress", "it is fine", "normal quality",
 "the dress is acceptable", "not bad not great", "reasonable product", "looks fine"
]
TRAIN_LABELS = ["positive"]*12 + ["negative"]*10 + ["neutral"]*8
ml_model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=500)), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
ml_model.fit(TRAIN_TEXTS, TRAIN_LABELS)

def get_api_key():
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    try:
        if not key and "YOUTUBE_API_KEY" in st.secrets:
            key = st.secrets["YOUTUBE_API_KEY"].strip()
    except Exception:
        pass
    return key

def fetch_youtube(query, max_videos, max_comments, api_key, fetch_all_comments=False, progress_callback=None):
    """Collect paginated public YouTube search results, video statistics and comments.

    Search results are paged in batches of up to 50. Comments are collected in
    batches of up to 100. fetch_all_comments=True follows every available page
    of top-level comments for each video (subject to API quota and availability).
    """
    youtube = build("youtube", "v3", developerKey=api_key)

    # YouTube search.list returns at most 50 results per request.
    ids = []
    seen_ids = set()
    page_token = None
    while len(ids) < max_videos:
        page_size = min(50, max_videos - len(ids))
        request_args = dict(part="snippet", q=query, type="video", regionCode="PK",
                            relevanceLanguage="ur", maxResults=page_size, order="relevance")
        if page_token:
            request_args["pageToken"] = page_token
        search = youtube.search().list(**request_args).execute()
        for item in search.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                ids.append(vid)
                if len(ids) >= max_videos:
                    break
        page_token = search.get("nextPageToken")
        if not page_token or not search.get("items"):
            break

    if not ids:
        return pd.DataFrame(), pd.DataFrame()

    # videos.list accepts up to 50 video IDs per request.
    video_items = []
    for i in range(0, len(ids), 50):
        video_resp = youtube.videos().list(part="snippet,statistics", id=",".join(ids[i:i+50])).execute()
        video_items.extend(video_resp.get("items", []))

    videos = []
    for v in video_items:
        sn, stt = v.get("snippet", {}), v.get("statistics", {})
        published = pd.to_datetime(sn.get("publishedAt"), utc=True, errors="coerce")
        title = sn.get("title", "")
        desc = sn.get("description", "")
        tags = sn.get("tags", [])
        hashtags = list(dict.fromkeys(re.findall(r"#([\w]+)", title + " " + desc)))
        videos.append({
            "video_id": v["id"], "title": title, "channel": sn.get("channelTitle", ""),
            "published_at": published, "date": published.date().isoformat() if pd.notna(published) else "",
            "day": published.day_name() if pd.notna(published) else "",
            "hour_utc": int(published.hour) if pd.notna(published) else None,
            "views": int(stt.get("viewCount", 0)), "likes": int(stt.get("likeCount", 0)),
            "comment_count": int(stt.get("commentCount", 0)), "hashtags": ", ".join(hashtags),
            "api_tags": ", ".join(tags), "url": "https://www.youtube.com/watch?v=" + v["id"],
            "description": desc[:1000]
        })

    comments = []
    titles_by_id = {v["video_id"]: v["title"] for v in videos}
    for video_index, vid in enumerate(ids, start=1):
        page_token = None
        collected_for_video = 0
        try:
            while fetch_all_comments or collected_for_video < max_comments:
                remaining = 100 if fetch_all_comments else min(100, max_comments - collected_for_video)
                if remaining <= 0:
                    break
                args = dict(part="snippet", videoId=vid, maxResults=remaining,
                            order="relevance", textFormat="plainText")
                if page_token:
                    args["pageToken"] = page_token
                resp = youtube.commentThreads().list(**args).execute()
                items = resp.get("items", [])
                for item in items:
                    thread_snippet = item.get("snippet", {})
                    top_comment = thread_snippet.get("topLevelComment", {}).get("snippet", {})
                    comments.append({"video_id": vid, "video_title": titles_by_id.get(vid, ""),
                                     "comment": top_comment.get("textDisplay", ""),
                                     "likes": int(top_comment.get("likeCount", 0)),
                                     "published_at": top_comment.get("publishedAt", "")})
                    collected_for_video += 1
                page_token = resp.get("nextPageToken")
                if not page_token or not items or (not fetch_all_comments and collected_for_video >= max_comments):
                    break
        except Exception:
            # Comments may be disabled or restricted for a video.
            pass
        if progress_callback:
            progress_callback(video_index, len(ids), len(comments))

    return pd.DataFrame(videos), pd.DataFrame(comments)

def sentiment_ml(text):
    return ml_model.predict([str(text)])[0]


@st.cache_resource
def load_dl_model():
    """Load the pretrained sentiment model once per Streamlit server session."""
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,  # This environment uses CPU-only PyTorch.
    )


@st.cache_data(show_spinner=False, max_entries=8)
def cached_dl_predictions(texts: tuple):
    """Cache sentiment predictions for an exact tuple of comment texts."""
    dl = load_dl_model()
    results = dl(
        list(texts),
        batch_size=4,  # A conservative starting point for CPU inference.
        truncation=True,
        max_length=128,
    )

    predictions = []
    for item in results:
        label = item["label"].upper()
        score = item["score"]
        if label == "POSITIVE" and score >= 0.60:
            predictions.append("positive")
        elif label == "NEGATIVE" and score >= 0.60:
            predictions.append("negative")
        else:
            predictions.append("neutral")
    return predictions

# Persistent dataset files. Collected YouTube data is saved here and reloaded on restart.
DATA_DIR = Path(__file__).resolve().parent / "data"
VIDEOS_CSV = DATA_DIR / "videos.csv"
COMMENTS_CSV = DATA_DIR / "comments.csv"

VIDEO_COLUMNS = [
    "video_id", "title", "channel", "published_at", "date", "day", "hour_utc",
    "views", "likes", "comment_count", "hashtags", "api_tags", "url", "description"
]
COMMENT_COLUMNS = ["video_id", "video_title", "comment", "likes", "published_at"]


def load_saved_data():
    """Load the most recently collected dataset, if it exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    videos = pd.read_csv(VIDEOS_CSV) if VIDEOS_CSV.exists() else pd.DataFrame(columns=VIDEO_COLUMNS)
    comments = pd.read_csv(COMMENTS_CSV) if COMMENTS_CSV.exists() else pd.DataFrame(columns=COMMENT_COLUMNS)

    # Restore numeric/date types that CSV storage may otherwise infer inconsistently.
    for column in ("views", "likes", "comment_count", "hour_utc"):
        if column in videos:
            videos[column] = pd.to_numeric(videos[column], errors="coerce")
    for column in ("likes",):
        if column in comments:
            comments[column] = pd.to_numeric(comments[column], errors="coerce").fillna(0).astype(int)
    if "published_at" in videos:
        videos["published_at"] = pd.to_datetime(videos["published_at"], errors="coerce", utc=True)
    return videos, comments


def save_collected_data(videos, comments):
    """Persist raw collected YouTube records for later app runs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    videos.to_csv(VIDEOS_CSV, index=False, encoding="utf-8-sig")
    comments.to_csv(COMMENTS_CSV, index=False, encoding="utf-8-sig")

with st.sidebar:
    st.header("YouTube data collection")
    st.caption("Collect real public YouTube data and save it in this project's data/ folder. Nothing is collected until you click the button.")
    query = st.text_input("Search topic", "Pakistani clothing")
    max_videos = st.slider(
        "Videos to collect", 50, 1000, 100, step=50,
        help="YouTube search results are paginated; the API may return fewer videos than requested."
    )
    fetch_all_comments = st.checkbox(
        "Fetch all available top-level comments per video", value=False,
        help="Can take a long time and consume API quota. Some videos disable comments."
    )
    max_comments = st.slider(
        "Comments per video (when not fetching all)", 10, 500, 100, step=10,
        disabled=fetch_all_comments
    )
    api_key_input = st.text_input(
        "Your YouTube API key", type="password",
        help="Use your own key, or set YOUTUBE_API_KEY in your local .env file. The app does not include a shared key."
    )
    load_button = st.button("Collect YouTube Data and Save", type="primary", use_container_width=True)

if "videos" not in st.session_state or "comments" not in st.session_state:
    saved_videos, saved_comments = load_saved_data()
    st.session_state.videos = saved_videos
    st.session_state.comments = saved_comments
    st.session_state.source_label = "Saved YouTube dataset" if not saved_videos.empty else "No dataset loaded"

if load_button:
    key = api_key_input.strip() or get_api_key()
    if not key:
        st.error("A YouTube Data API key is required. Enter your own key above or add YOUTUBE_API_KEY to your .env file. No API request was made.")
    else:
        try:
            with st.spinner("Collecting YouTube videos and public comments..."):
                progress = st.progress(0, text="Searching YouTube videos...")

                def update_progress(done, total, comment_count):
                    progress.progress(
                        min(done / max(total, 1), 1.0),
                        text=f"Collecting comments: video {done}/{total} • {comment_count:,} comments"
                    )

                collected_videos, collected_comments = fetch_youtube(
                    query, max_videos, max_comments, key,
                    fetch_all_comments=fetch_all_comments,
                    progress_callback=update_progress
                )
                progress.empty()

            if collected_videos.empty:
                st.warning("YouTube returned no videos. Try another search topic or check your API key and quota.")
            else:
                save_collected_data(collected_videos, collected_comments)
                st.session_state.videos = collected_videos
                st.session_state.comments = collected_comments
                st.session_state.source_label = "Live YouTube Data API (saved in data/)"
                # New data should not reuse the previous dataset's DL display state.
                st.session_state.run_dl = False
                st.session_state.sentiment_result = "ML baseline only"
                st.success(
                    f"Saved {len(collected_videos):,} videos and {len(collected_comments):,} comments to "
                    f"{DATA_DIR.relative_to(Path(__file__).resolve().parent)}/videos.csv and comments.csv."
                )
        except Exception as e:
            st.error(f"YouTube collection failed: {e}. Check API enablement, key restrictions, internet access, and quota.")

videos = st.session_state.videos.copy()
comments = st.session_state.comments.copy()

if videos.empty:
    st.warning("No YouTube dataset is saved yet. Enter your own YouTube Data API key in the sidebar, choose a search topic, and click **Collect YouTube Data and Save**. The app will then analyze the saved dataset.")
    st.info(f"Expected dataset files: `{VIDEOS_CSV}` and `{COMMENTS_CSV}`")
    st.stop()
if not comments.empty:
    # ML labels are lightweight and can be recreated on every run.
    comments["sentiment_ml"] = comments["comment"].apply(sentiment_ml)

    # Keep previously saved DistilBERT predictions from data/comments.csv.
    # Only mark the column as "Not run" when no valid saved predictions exist.
    if "sentiment_dl" not in comments.columns:
        comments["sentiment_dl"] = "Not run"
    else:
        comments["sentiment_dl"] = comments["sentiment_dl"].fillna("Not run").astype(str)

    valid_dl_labels = {"positive", "negative", "neutral"}
    has_saved_dl = comments["sentiment_dl"].str.lower().isin(valid_dl_labels).all()
    if not has_saved_dl:
        comments.loc[~comments["sentiment_dl"].str.lower().isin(valid_dl_labels), "sentiment_dl"] = "Not run"

    if st.session_state.get("run_dl", False) and (not has_saved_dl or st.session_state.get("force_dl", False)):
        try:
            texts = tuple(comments["comment"].fillna("").astype(str).tolist())
            with st.spinner(
                f"Loading/reusing DistilBERT and analyzing {len(texts):,} comments. "
                "First run may take several minutes on CPU..."
            ):
                comments["sentiment_dl"] = cached_dl_predictions(texts)
            # Persist model output so it appears automatically after app restart / URL reopen.
            save_collected_data(videos, comments)
            has_saved_dl = True
            st.session_state.sentiment_result = "Pretrained Transformer + ML baseline (saved)"
        except Exception as e:
            st.warning(f"DL model unavailable; showing ML results. {e}")
            st.session_state.sentiment_result = "ML baseline only (DL unavailable)"
        finally:
            st.session_state.run_dl = False
            st.session_state.force_dl = False

    if has_saved_dl:
        st.session_state.sentiment_result = "Pretrained Transformer + ML baseline (saved)"
    else:
        st.session_state.sentiment_result = "ML baseline only"
else:
    comments = pd.DataFrame(columns=[
        "video_id", "video_title", "comment", "likes", "published_at",
        "sentiment_ml", "sentiment_dl"
    ])

tabs = st.tabs(["📊 Overview", "🎬 Video Analysis", "💬 Sentiment NLP", "🏷️ Hashtags & Timing", "💡 Recommendations", "🤖 Data Chat", "📥 Export"])

with tabs[0]:
    st.caption(f"Data source: **{st.session_state.source_label}**")
    st.caption(f"Dataset: **{len(videos):,} saved YouTube videos** and **{len(comments):,} saved comments**.")
    a,b,c,d = st.columns(4)
    a.metric("Videos analysed", f"{len(videos):,}")
    b.metric("Total views", f"{int(videos.views.sum()):,}")
    c.metric("Total likes", f"{int(videos.likes.sum()):,}")
    d.metric("Comments collected", f"{len(comments):,}")
    videos["engagement_rate"] = ((videos["likes"] + videos["comment_count"]) / videos["views"].replace(0,1) * 100).round(2)
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(videos.nlargest(10,"views"), x="title", y="views", title="Top videos by views"), use_container_width=True)
    with right:
        st.plotly_chart(px.scatter(videos, x="views", y="likes", size="comment_count", hover_name="title", title="Views vs likes (bubble size = comments)"), use_container_width=True)
    if not comments.empty:
        counts = comments["sentiment_ml"].value_counts().rename_axis("sentiment").reset_index(name="comments")
        st.plotly_chart(px.pie(counts, names="sentiment", values="comments", title="Comment sentiment — ML baseline"), use_container_width=True)
    st.dataframe(videos[["title","channel","date","day","hour_utc","views","likes","comment_count","engagement_rate","url"]], use_container_width=True, hide_index=True)

with tabs[1]:
    videos["engagement_rate"] = ((videos["likes"] + videos["comment_count"]) / videos["views"].replace(0,1) * 100).round(2)
    metric = st.selectbox("Rank videos by", ["views","likes","comment_count","engagement_rate"])
    st.dataframe(videos.sort_values(metric, ascending=False)[["title","channel","date","day","hour_utc",metric,"url"]], use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(videos.sort_values(metric, ascending=False), x="title", y=metric, title=f"Video ranking by {metric.replace('_',' ').title()}"), use_container_width=True)

with tabs[2]:
    st.subheader("Sentiment analysis: ML + DL + NLP")
    st.write("NLP step: text is processed by TF-IDF features for the traditional ML baseline. The ML model is trained on a small built-in starter set; validate it with a labelled local dataset before making business decisions.")
    if comments.empty:
        st.warning("No public comments were collected for these videos. Try videos with comments enabled or collect another dataset.")
    else:
        dl_is_saved = comments["sentiment_dl"].astype(str).str.lower().isin({"positive", "negative", "neutral"}).all()
        if dl_is_saved:
            st.success("Saved DistilBERT sentiment results loaded from data/comments.csv.")
            if st.button("Re-run pretrained DL sentiment model"):
                st.session_state.run_dl = True
                st.session_state.force_dl = True
                # Force a fresh inference by clearing the prediction cache.
                cached_dl_predictions.clear()
                st.rerun()
        else:
            if st.button("Run pretrained DL sentiment model"):
                st.session_state.run_dl = True
                st.rerun()
        st.caption("DistilBERT results are saved in data/comments.csv after the first successful run and reused when the app is reopened.")
        chosen = "sentiment_dl" if dl_is_saved else "sentiment_ml"
        st.write(f"Models displayed: **{st.session_state.get('sentiment_result','ML baseline only')}**")
        counts = comments[chosen].value_counts().rename_axis("sentiment").reset_index(name="comments")
        st.plotly_chart(px.bar(counts, x="sentiment", y="comments", color="sentiment", title="Sentiment distribution"), use_container_width=True)
        st.dataframe(comments[["video_title","comment","sentiment_ml"] + (["sentiment_dl"] if "sentiment_dl" in comments else [])], use_container_width=True, hide_index=True)
        st.download_button("Download sentiment CSV", comments.to_csv(index=False).encode("utf-8"), "sentiment_results.csv", "text/csv")

with tabs[3]:
    st.subheader("Hashtags and posting time")
    tag_counter = Counter()
    for text in videos.get("hashtags", pd.Series(dtype=str)).fillna(""):
        for tag in str(text).split(","):
            if tag.strip(): tag_counter["#"+tag.strip().lstrip("#")] += 1
    if tag_counter:
        tags_df = pd.DataFrame(tag_counter.most_common(15), columns=["hashtag","video_count"])
        st.plotly_chart(px.bar(tags_df, x="hashtag", y="video_count", title="Most common hashtags in collected videos"), use_container_width=True)
        st.dataframe(tags_df, hide_index=True, use_container_width=True)
    else:
        st.info("No hashtags were found in video titles/descriptions. YouTube API tags are not always public.")
    valid = videos.dropna(subset=["hour_utc"]).copy()
    if not valid.empty:
        by_hour = valid.groupby("hour_utc", as_index=False).agg(avg_views=("views","mean"), videos=("video_id","count"))
        st.plotly_chart(px.bar(by_hour, x="hour_utc", y="avg_views", hover_data=["videos"], title="Average views by publication hour (UTC)"), use_container_width=True)
    by_day = videos.groupby("day", as_index=False).agg(avg_views=("views","mean"), videos=("video_id","count"))
    if not by_day.empty:
        st.plotly_chart(px.bar(by_day, x="day", y="avg_views", hover_data=["videos"], title="Average views by weekday"), use_container_width=True)
    st.warning("YouTube publication timestamps are UTC. Convert to Pakistan Standard Time (UTC+5) before scheduling. Observational correlations do not prove the best time to post.")

with tabs[4]:
    st.subheader("AI-assisted marketing recommendations")
    st.write(
        "Recommendations are generated from patterns in your collected YouTube videos, "
        "engagement metrics, hashtags, publishing times, and available comment sentiment. "
        "They are evidence-based suggestions to test, not guaranteed predictions."
    )

    if videos.empty:
        st.info("Collect YouTube data first to generate recommendations.")
    else:
        videos = videos.copy()
        videos["views"] = pd.to_numeric(videos.get("views", 0), errors="coerce").fillna(0)
        videos["likes"] = pd.to_numeric(videos.get("likes", 0), errors="coerce").fillna(0)
        videos["comment_count"] = pd.to_numeric(videos.get("comment_count", 0), errors="coerce").fillna(0)
        videos["engagement_rate"] = (
            (videos["likes"] + videos["comment_count"]) /
            videos["views"].replace(0, 1) * 100
        ).round(2)

        top_views = videos.loc[videos["views"].idxmax()]
        top_engagement = videos.loc[videos["engagement_rate"].idxmax()]
        avg_views = float(videos["views"].mean()) if len(videos) else 0.0
        avg_engagement = float(videos["engagement_rate"].mean()) if len(videos) else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Videos in dataset", f"{len(videos):,}")
        m2.metric("Average views", f"{avg_views:,.0f}")
        m3.metric("Average engagement proxy", f"{avg_engagement:.2f}%")

        st.markdown("### Patterns discovered in your dataset")
        st.markdown(
            f"- **Reach pattern:** `{top_views['title']}` has the most views "
            f"({int(top_views['views']):,}). Review its product, title, thumbnail, and opening seconds for ideas to test."
        )
        st.markdown(
            f"- **Engagement pattern:** `{top_engagement['title']}` has the highest engagement proxy "
            f"({top_engagement['engagement_rate']:.2f}%). Compare its format and audience response with lower-engagement videos."
        )

        # Use the saved DistilBERT predictions when available; otherwise use the ML baseline.
        sentiment_col = None
        if not comments.empty:
            if "sentiment_dl" in comments.columns and comments["sentiment_dl"].astype(str).str.lower().isin({"positive", "negative", "neutral"}).all():
                sentiment_col = "sentiment_dl"
            elif "sentiment_ml" in comments.columns:
                sentiment_col = "sentiment_ml"

        negative_share = None
        if sentiment_col and not comments.empty:
            sentiment_counts = comments[sentiment_col].astype(str).str.lower().value_counts()
            total_sentiments = int(sentiment_counts.sum())
            if total_sentiments:
                positive_share = 100 * sentiment_counts.get("positive", 0) / total_sentiments
                negative_share = 100 * sentiment_counts.get("negative", 0) / total_sentiments
                neutral_share = 100 * sentiment_counts.get("neutral", 0) / total_sentiments
                st.markdown(
                    f"- **Audience sentiment ({sentiment_col.replace('sentiment_', '').upper()}):** "
                    f"{positive_share:.1f}% positive, {negative_share:.1f}% negative, "
                    f"and {neutral_share:.1f}% neutral across {total_sentiments:,} comments."
                )
                if negative_share >= 30:
                    st.warning(
                        "A relatively high share of comments is labelled negative. Read a sample of negative comments "
                        "before changing your campaign; the current sentiment model may misclassify Urdu or Roman Urdu."
                    )
                elif positive_share >= 60:
                    st.success(
                        "Positive sentiment dominates this dataset. Test more content around the products or presentation "
                        "that receive favourable responses, while checking the actual comments for context."
                    )
                else:
                    st.info(
                        "Sentiment is mixed. Test clearer product details, pricing, sizing, fabric close-ups, and delivery information; "
                        "use the comments to identify which issue matters most."
                    )

        st.markdown("### Recommended marketing actions")
        actions = []

        # Hashtag patterns grounded in collected video metadata.
        if "hashtags" in videos.columns:
            hashtag_counter = Counter()
            for raw_tags in videos["hashtags"].fillna(""):
                for tag in str(raw_tags).split(","):
                    tag = tag.strip().lstrip("#")
                    if tag:
                        hashtag_counter[tag] += 1
            if hashtag_counter:
                popular_tags = [f"#{tag}" for tag, _ in hashtag_counter.most_common(5)]
                actions.append((
                    "Build a relevant hashtag test",
                    f"The most frequent hashtags in this dataset are {', '.join(popular_tags)}. Test a small, relevant set on future posts and compare results; frequency here does not prove that a hashtag caused higher views."
                ))

        # Publication time is in UTC in the collected dataset.
        if "hour_utc" in videos.columns:
            timed = videos.dropna(subset=["hour_utc"]).copy()
            if not timed.empty:
                timed["hour_utc"] = pd.to_numeric(timed["hour_utc"], errors="coerce")
                timed = timed.dropna(subset=["hour_utc"])
                if not timed.empty:
                    best_hour = int(timed.groupby("hour_utc")["views"].mean().idxmax())
                    actions.append((
                        "Run a posting-time experiment",
                        f"Videos posted around {best_hour:02d}:00 UTC ({(best_hour + 5) % 24:02d}:00 Pakistan time) have the highest average views in this sample. Schedule comparable posts at this time and at one or two alternative times to validate the pattern."
                    ))

        # Infer common customer concerns from negative comments using simple keyword matching.
        if not comments.empty and "comment" in comments.columns:
            if sentiment_col:
                negative_comments = comments.loc[
                    comments[sentiment_col].astype(str).str.lower().eq("negative"), "comment"
                ].fillna("").astype(str)
            else:
                negative_comments = pd.Series(dtype=str)
            concern_terms = {
                "price or affordability": ["price", "expensive", "cost", "مہنگا", "قیمت", "mehnga"],
                "fabric or quality": ["quality", "fabric", "material", "خراب", "کوالٹی", "kapra", "kapray"],
                "stitching or fit": ["stitch", "stitching", "size", "fit", "سلائی", "size"],
                "delivery or service": ["delivery", "late", "service", "shipping", "ارسال"]
            }
            lower_negative = negative_comments.str.lower()
            concern_counts = {
                concern: int(sum(lower_negative.str.contains(re.escape(term), regex=True, na=False).sum() for term in terms))
                for concern, terms in concern_terms.items()
            }
            concern_counts = {k: v for k, v in concern_counts.items() if v > 0}
            if concern_counts:
                leading_concern, hits = max(concern_counts.items(), key=lambda item: item[1])
                actions.append((
                    "Address a possible customer concern",
                    f"Keyword matching found the most mentions related to **{leading_concern}** in comments labelled negative ({hits} keyword hits, which may overlap). Consider answering this concern clearly in product videos and captions, then check a sample of comments manually."
                ))

        # Data-specific creative suggestion based on top-viewed title.
        actions.append((
            "Create a follow-up to the strongest-reach video",
            f"Use `{top_views['title']}` as a reference for a new post: keep the relevant product/category theme, but create a fresh short video showing fabric texture, close-up stitching, available sizes, price range, and ordering/delivery details. Compare views and engagement with the original."
        ))

        for index, (heading, detail) in enumerate(actions, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {heading}**")
                st.write(detail)

        st.caption(
            "Method note: recommendations use descriptive patterns in the collected sample, simple keyword matching, "
            "and the available sentiment labels. They do not establish causation. The built-in ML starter model is small, "
            "and the English DistilBERT model may be unreliable on Urdu/Roman Urdu comments. Validate with labelled local data."
        )

with tabs[5]:
    st.subheader("Ask the dashboard")
    question = st.text_input("Ask about this dataset", placeholder="Which video has the most views?")
    if st.button("Ask"):
        q = question.lower()
        if any(w in q for w in ["most view","highest view","top video"]):
            r = videos.loc[videos["views"].idxmax()]
            st.write(f"**Top by views:** {r['title']} — {int(r['views']):,} views.")
        elif "like" in q:
            r = videos.loc[videos["likes"].idxmax()]
            st.write(f"**Top by likes:** {r['title']} — {int(r['likes']):,} likes.")
        elif "sentiment" in q or "positive" in q or "negative" in q:
            if comments.empty: st.write("No comments are available to analyse.")
            else: st.write(comments["sentiment_ml"].value_counts().to_dict())
        elif "hashtag" in q or "tag" in q:
            st.write("Most common hashtags: " + ", ".join("#"+t.lstrip("#") for t,n in tag_counter.most_common(8)))
        elif "time" in q or "hour" in q or "day" in q:
            if videos["hour_utc"].notna().any():
                h = int(videos.groupby("hour_utc")["views"].mean().idxmax())
                st.write(f"Highest average views in this sample: {h:02d}:00 UTC / {(h+5)%24:02d}:00 Pakistan time.")
            else: st.write("Publication-time data is unavailable.")
        else:
            st.write(f"This dataset contains {len(videos)} videos, {int(videos.views.sum()):,} total views, and {len(comments)} collected comments. Try asking about top video, likes, sentiment, hashtags, or posting time.")

with tabs[6]:
    st.subheader("Export data")
    st.download_button("Download video analytics CSV", videos.to_csv(index=False).encode("utf-8"), "youtube_video_analytics.csv", "text/csv")
    if not comments.empty:
        st.download_button("Download comments + sentiment CSV", comments.to_csv(index=False).encode("utf-8"), "youtube_comments_sentiment.csv", "text/csv")
    st.write("For your report, state the date of collection, search query, number of videos/comments, and the limitations of public API data.")
