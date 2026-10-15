# Social Media Engagement Analyzer 📱

**Day 8 of the 15-day Daily Data-AI GitHub series** — find the best time,
day, and format to post, and predict how engaging a post will be.

## What it does

1. **Generates** 1,000 synthetic social-media posts (Jan–Jun 2026) with a
   fixed random seed — `generate_data.py`. No external APIs, fully
   reproducible.
2. **Analyzes** engagement by hour of day, day of week, content format,
   and hashtag count — 4 EDA charts in `charts/`.
3. **Predicts** engagement rate with a `RandomForestRegressor`
   (R² ≈ 0.68 on held-out posts) and ranks the most important drivers —
   `actual_vs_predicted.png`, `feature_importance.png`.
4. **Explains** the findings in plain English — `ai_insights.py` writes
   `AI_INSIGHTS.md`: headline findings + 5 concrete content tips.

## Key findings

- Post between **7 PM and 9 PM** for the highest engagement.
- **Saturday** is the best day; Monday the worst.
- **Reels win**: 6.27% avg engagement vs 2.80% for text posts (~124% lift).
- **3–6 hashtags** is the sweet spot; 11+ looks spammy and hurts reach.
- The model says *when* and *what* you post matter far more than caption
  length or mention counts.

## How to run

```bash
pip install -r requirements.txt
python social_media_engagement.py   # EDA + charts + model -> findings.json
python ai_insights.py               # plain-English report -> AI_INSIGHTS.md
```

The dataset (`data/posts.csv`) is committed and small (~70 KB). To
regenerate it from scratch, delete it and run `generate_data.py` —
the fixed seed (`42`) reproduces identical data.

## The "AI integrated" part

The `ai_insights.py` module turns the model's numbers into an automated
narrative report — findings, why they happened, and actionable posting
recommendations. That's the AI layer: analysis that writes its own summary.
