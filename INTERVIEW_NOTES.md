# Interview Notes — Social Media Engagement Analyzer

## 30-second pitch

"I built a social-media analytics tool on 1,000 synthetic posts. It finds
the best time, day, and format to post — 7–9 PM, Saturday, Reels — then
trains a Random Forest that predicts a post's engagement rate with R² of
0.68. An insights module auto-writes a plain-English report with five
actionable content tips."

## Project walkthrough (end-to-end)

1. `generate_data.py` creates 1,000 posts with timestamps, formats, caption
   lengths, hashtag counts, follower counts, and likes/comments/shares —
   seeded RNG (`42`) so it's reproducible. Synthetic data was used because
   real account data wasn't available, and it lets the analysis be shared
   publicly without privacy concerns.
2. `social_media_engagement.py` loads the CSV, groups engagement by hour /
   day / format / hashtag bucket into four bar charts, then trains a
   RandomForestRegressor (200 trees, max depth 10) on post features to
   predict engagement rate. It saves actual-vs-predicted and feature
   importance charts plus a machine-readable `findings.json`.
3. `ai_insights.py` reads `findings.json` and generates `AI_INSIGHTS.md` —
   the narrative layer.

## Technical deep-dives

**Engagement rate.** Defined as (likes + comments + shares) / reach, not per
follower. This is the honest metric: followers don't all see a post, so
normalizing by reach measures how the content itself performed. Limitation:
reach here is simulated, so the denominator is as synthetic as the numerator.

**Why Random Forest for regression?** Engagement depends on interacting,
non-linear factors (format × hour × day). A single linear regression would
miss those interactions; a forest captures them with little tuning and gives
feature importances for free. Limitation: importances are biased toward
high-cardinality features (hour has 24 values, format dummies have 2), so
hour can look more important partly because of granularity, not just signal.

**One-hot encoding.** Day of week and format are categorical, so they're
converted to 0/1 dummy columns — trees can't split on "Reel vs Image" text
directly. Limitation: this expands 2 columns into 11; with tiny datasets
that sparsity can hurt, though 1,000 rows is fine.

**R² = 0.68, MAE = 0.012.** R² means the model explains 70% of the variance
in engagement rate — solid for behavioral data. MAE of 0.012 means
predictions are off by about 1.2 percentage points on average against a
4.9% baseline. Limitation: the remaining 30% is unmeasured stuff — topic
quality, trending audio, creator skill — which no feature set here captures.

**Hashtag bucketing.** Raw counts (0–15) were binned into 0–2 / 3–6 / 7–10 /
11+ for a readable chart. Limitation: binning hides within-bin structure and
the cut points are judgment calls; a different binning could shift the story.

## Data-analyst thinking

KPIs tracked: average engagement rate (4.70%), per-segment rates by time,
day, format, and hashtag use, plus model error metrics. The key insight is
that *distribution timing and format dominate everything else* — a Reel at
8 PM outperforms a text post at 3 AM by roughly 3×. The analysis became
recommendations by pairing each finding with an action: best hours → posting
schedule, best format → repurpose images into Reels, hashtag sweet spot →
cap tags at six. That translation from numbers to "do this on Saturday" is
the analyst's job.

## The "AI integrated" part

"AI integrated" here means the pipeline doesn't just output numbers — the
`ai_insights.py` module reads the JSON findings and writes the executive
summary itself: findings, causal interpretation, and five prioritized tips.
Trade-off vs alternatives: a template-based generator is fully deterministic
and cheap but can't reason about surprising results the way an LLM could;
an LLM-written report would be more fluent but non-deterministic and
costlier. For a reproducible portfolio project, determinism won.

## With more time I would…

Add topic/content features (sentiment of caption, presence of trending audio)
since content quality is the biggest unexplained variance. Try a gradient
boosting model and tune it with cross-validation instead of a single
train/test split. Add a "best combination" heatmap (format × hour) and a
simple simulator: "given this post plan, expected engagement is X."
