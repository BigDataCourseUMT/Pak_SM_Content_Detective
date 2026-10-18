import os, re
from datetime import datetime
from collections import Counter
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

load_dotenv()
st.set_page_config(page_title="Pakistani Social Media Content Detective", page_icon="🕵️", layout="wide")

st.markdown("""
<style>
.main {background-color:#f7f9fc}
[data-testid="stMetric"] {background:white;padding:14px;border-radius:12px;border:1px solid #e7ebf2}
h1,h2,h3 {color:#12304a}
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Pakistani Social Media Content Detective")
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

def fetch_youtube(query, max_videos, max_comments, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    search = youtube.search().list(part="snippet", q=query, type="video", regionCode="PK",
                                   relevanceLanguage="ur", maxResults=max_videos, order="relevance").execute()
    ids = [x["id"]["videoId"] for x in search.get("items", [])]
    if not ids:
        return pd.DataFrame(), pd.DataFrame()
    video_resp = youtube.videos().list(part="snippet,statistics", id=",".join(ids)).execute()
    videos = []
    for v in video_resp.get("items", []):
        sn, stt = v.get("snippet", {}), v.get("statistics", {})
        published = pd.to_datetime(sn.get("publishedAt"), utc=True, errors="coerce")
        title = sn.get("title", "")
        desc = sn.get("description", "")
        tags = sn.get("tags", [])
        hashtags = list(dict.fromkeys(re.findall(r"#([\w]+)", title + " " + desc)))
        videos.append({
            "video_id": v["id"], "title": title, "channel": sn.get("channelTitle",""),
            "published_at": published, "date": published.date().isoformat() if pd.notna(published) else "",
            "day": published.day_name() if pd.notna(published) else "",
            "hour_utc": int(published.hour) if pd.notna(published) else None,
            "views": int(stt.get("viewCount",0)), "likes": int(stt.get("likeCount",0)),
            "comment_count": int(stt.get("commentCount",0)), "hashtags": ", ".join(hashtags),
            "api_tags": ", ".join(tags), "url": "https://www.youtube.com/watch?v="+v["id"],
            "description": desc[:1000]
        })
    comments = []
    for vid in ids:
        try:
            resp = youtube.commentThreads().list(part="snippet", videoId=vid, maxResults=min(max_comments,100),
                                                 order="relevance", textFormat="plainText").execute()
            title = next((x["title"] for x in videos if x["video_id"] == vid), "")
            for item in resp.get("items", []):
                c = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({"video_id":vid, "video_title":title, "comment":c.get("textDisplay",""),
                                 "likes":int(c.get("likeCount",0)), "published_at":c.get("publishedAt","")})
        except Exception:
            # Comments may be disabled or restricted for a video.
            continue
    return pd.DataFrame(videos), pd.DataFrame(comments)

def sentiment_ml(text):
    return ml_model.predict([str(text)])[0]

def make_demo_data():
    videos = pd.DataFrame([
      {"video_id":"demo1","title":"Eid Collection | Pakistani Lawn Suit #EidCollection #Lawn","channel":"Demo Clothing PK","published_at":pd.Timestamp("2026-09-10T14:00:00Z"),"date":"2026-09-10","day":"Thursday","hour_utc":14,"views":18000,"likes":900,"comment_count":130,"hashtags":"EidCollection, Lawn","api_tags":"eid, lawn","url":"https://www.youtube.com","description":"Demo data only"},
      {"video_id":"demo2","title":"Winter Khaddar Collection #WinterWear #PakistaniFashion","channel":"Demo Fashion","published_at":pd.Timestamp("2026-09-12T17:00:00Z"),"date":"2026-09-12","day":"Saturday","hour_utc":17,"views":24000,"likes":1200,"comment_count":180,"hashtags":"WinterWear, PakistaniFashion","api_tags":"winter, khaddar","url":"https://www.youtube.com","description":"Demo data only"},
      {"video_id":"demo3","title":"New Embroidered Suit Designs #Embroidered #PretWear","channel":"Demo Boutique","published_at":pd.Timestamp("2026-09-14T12:00:00Z"),"date":"2026-09-14","day":"Monday","hour_utc":12,"views":9500,"likes":330,"comment_count":70,"hashtags":"Embroidered, PretWear","api_tags":"pret, embroidery","url":"https://www.youtube.com","description":"Demo data only"},
      {"video_id":"demo4","title":"Affordable Unstitched Collection #Unstitched #SummerLawn","channel":"Demo Clothing PK","published_at":pd.Timestamp("2026-09-16T15:00:00Z"),"date":"2026-09-16","day":"Wednesday","hour_utc":15,"views":14500,"likes":700,"comment_count":95,"hashtags":"Unstitched, SummerLawn","api_tags":"summer, lawn","url":"https://www.youtube.com","description":"Demo data only"}
    ])
    comments = pd.DataFrame([
      {"video_id":"demo1","video_title":"Eid Collection","comment":"Beautiful suit and lovely colors","likes":8,"published_at":""},
      {"video_id":"demo1","video_title":"Eid Collection","comment":"Price is too high for this quality","likes":3,"published_at":""},
      {"video_id":"demo2","video_title":"Winter Khaddar","comment":"Amazing collection, fabric looks warm","likes":10,"published_at":""},
      {"video_id":"demo2","video_title":"Winter Khaddar","comment":"Delivery was late and disappointing","likes":2,"published_at":""},
      {"video_id":"demo3","video_title":"Embroidered Suit","comment":"Nice design, looks good","likes":4,"published_at":""},
      {"video_id":"demo4","video_title":"Summer Lawn","comment":"Average quality but reasonable price","likes":1,"published_at":""}
    ])
    return videos, comments

with st.sidebar:
    st.header("⚙️ Data settings")
    mode = st.radio("Choose data source", ["Demo data (works immediately)", "YouTube Data API"])
    query = st.text_input("Search topic", "Pakistani clothing brand lawn collection")
    max_videos = st.slider("Videos to collect", 5, 25, 10)
    max_comments = st.slider("Comments per video", 10, 100, 30, step=10)
    api_key_input = st.text_input("YouTube API key (optional)", type="password", help="Can also be saved as YOUTUBE_API_KEY in .env")
    load_button = st.button("🔎 Collect / Load Data", type="primary", use_container_width=True)

if "videos" not in st.session_state:
    st.session_state.videos, st.session_state.comments = make_demo_data()
    st.session_state.source_label = "Demo data (illustrative)"
if load_button:
    if mode.startswith("Demo"):
        st.session_state.videos, st.session_state.comments = make_demo_data()
        st.session_state.source_label = "Demo data (illustrative)"
    else:
        key = api_key_input.strip() or get_api_key()
        if not key:
            st.error("Enter a YouTube API key in the sidebar or add YOUTUBE_API_KEY to your .env file.")
        else:
            try:
                with st.spinner("Collecting public video statistics and available comments..."):
                    v, c = fetch_youtube(query, max_videos, max_comments, key)
                if v.empty:
                    st.warning("No videos were returned. Try a broader search phrase.")
                else:
                    st.session_state.videos, st.session_state.comments = v, c
                    st.session_state.source_label = "Live YouTube Data API"
                    st.success(f"Collected {len(v)} videos and {len(c)} available comments.")
            except Exception as e:
                st.error(f"YouTube request failed: {e}. Check API enablement, key restrictions, and quota.")

videos = st.session_state.videos.copy()
comments = st.session_state.comments.copy()
if videos.empty:
    st.warning("No data loaded yet.")
    st.stop()
if not comments.empty:
    comments["sentiment_ml"] = comments["comment"].apply(sentiment_ml)
    comments["sentiment_dl"] = "Not run"
    if "sentiment_result" not in st.session_state:
        st.session_state.sentiment_result = "ML baseline only"
    if st.session_state.get("run_dl", False):
        try:
            from transformers import pipeline
            @st.cache_resource
            def load_dl_model():
                return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            with st.spinner("Loading pretrained deep-learning sentiment model (first run downloads model)..."):
                dl = load_dl_model()
            def dl_sentiment(txt):
                result = dl(str(txt)[:512])[0]
                if result["label"].upper() == "POSITIVE" and result["score"] >= .60: return "positive"
                if result["label"].upper() == "NEGATIVE" and result["score"] >= .60: return "negative"
                return "neutral"
            comments["sentiment_dl"] = comments["comment"].apply(dl_sentiment)
            st.session_state.sentiment_result = "Pretrained Transformer + ML baseline"
        except Exception as e:
            st.warning(f"DL model unavailable; showing ML results. {e}")
            st.session_state.sentiment_result = "ML baseline only (DL unavailable)"
else:
    comments = pd.DataFrame(columns=["video_id","video_title","comment","likes","published_at","sentiment_ml","sentiment_dl"])

tabs = st.tabs(["📊 Overview", "🎬 Video Analysis", "💬 Sentiment NLP", "🏷️ Hashtags & Timing", "💡 Recommendations", "🤖 Data Chat", "📥 Export"])

with tabs[0]:
    st.caption(f"Data source: **{st.session_state.source_label}**")
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
        st.warning("No comments were returned. Use demo data or try videos with public comments enabled.")
    else:
        if st.button("Run pretrained DL sentiment model"):
            st.session_state.run_dl = True
            st.rerun()
        st.caption("DL option uses a pretrained DistilBERT Transformer. First run requires internet and may take time.")
        chosen = "sentiment_dl" if st.session_state.get("run_dl", False) and comments["sentiment_dl"].ne("Not run").any() else "sentiment_ml"
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
    st.subheader("Recommended next content")
    videos["engagement_rate"] = ((videos["likes"] + videos["comment_count"]) / videos["views"].replace(0,1) * 100).round(2)
    best = videos.sort_values("engagement_rate", ascending=False).iloc[0]
    st.success(f"Strongest observed engagement in this sample: **{best['title']}** ({best['engagement_rate']:.2f}% engagement proxy).")
    st.write("**Content ideas to test**")
    st.markdown("- Short close-up video showing fabric, stitching, and embroidery.\n- Show the price range, available sizes, and delivery details clearly.\n- Create an Eid, wedding-season, lawn, khaddar, or pret collection video that matches the brand's actual products.\n- Use a clear call to action: ask viewers which colour or design they prefer.")
    if tag_counter:
        st.write("**Hashtags found in this sample**")
        st.code(" ".join(["#"+t.lstrip("#") for t,n in tag_counter.most_common(8)]))
    st.write("**Posting-time suggestion**")
    if not videos.dropna(subset=["hour_utc"]).empty:
        best_hour = int(videos.groupby("hour_utc")["views"].mean().idxmax())
        st.write(f"The sample's highest average views occurred around {best_hour:02d}:00 UTC ({(best_hour+5)%24:02d}:00 Pakistan time). Test this time against other slots; the sample may be small or affected by video age.")
    st.write("**Ad concept template**")
    st.text_area("Draft ad idea", "Show the new collection in a 15–25 second vertical video. Start with the strongest design, show fabric detail and price, then end with WhatsApp/order instructions.", height=100)
    st.caption("Recommendations are practical hypotheses, not guaranteed outcomes. YouTube search results may include creators and brands from outside Pakistan.")

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
