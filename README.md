# Pakistani Social Media Content Detective

A Streamlit dashboard that studies public YouTube videos about Pakistani clothing/fashion brands. It includes video performance analytics, hashtags, publication day/hour analysis, comment sentiment, content ideas, a simple dataset chatbot, and CSV exports.

## Features
- YouTube Data API collection for video titles, channels, publication date, views, likes, comment counts, descriptions, and available hashtags/tags.
- Traditional ML sentiment baseline using TF-IDF + Logistic Regression.
- Optional pretrained deep-learning sentiment analysis using DistilBERT.
- Dashboard charts and CSV downloads.
- Heuristic recommendations for content, hashtags, and posting time.
- Demo data mode so the dashboard works without an API key.

## Run in VS Code (Windows)
1. Install Python 3.11 or 3.12.
2. Open this folder in VS Code.
3. In the VS Code terminal run:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`, then add your YouTube Data API key.
5. Run:
   ```powershell
   streamlit run app.py
   ```
6. Choose Demo data first to check the dashboard. Then choose YouTube Data API and click Collect / Load Data.

## Get a YouTube API key
Open Google Cloud Console, create/select a project, enable **YouTube Data API v3**, create an API key, and restrict it appropriately. Keep the key private. API quota and public comment availability can limit results.

## Deploy
Push the project to GitHub without `.env`. On Streamlit Community Cloud, deploy `app.py` from the repository and add this secret in Advanced settings:
```toml
YOUTUBE_API_KEY = "your-key-here"
```

## Important limitations
- Public API statistics do not expose a normal organic share-count field.
- Comments can be disabled or restricted.
- Video publication times are returned in UTC; Pakistan Standard Time is UTC+5.
- Hashtags are extracted from titles/descriptions. API tags may be absent.
- The built-in labelled examples are only a demonstration baseline, not a validated research dataset. Its predictions can be inaccurate, especially for Urdu, Pashto, and Roman Urdu.
- DistilBERT's default sentiment model is English-oriented and should not be treated as reliable for Urdu/Pashto. For a research-quality system, evaluate suitable multilingual models and a labelled local dataset.
- Posting-time recommendations are descriptive patterns, not causal predictions. A small sample may be biased by channel size, video age, and search ranking.

## Suggested academic evaluation
Label a balanced sample of comments as positive/negative/neutral, then report accuracy, precision, recall, F1-score, and a confusion matrix for the ML model and an appropriate multilingual DL model.
