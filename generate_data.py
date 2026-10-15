"""
generate_data.py
---------------
Generates a synthetic social-media dataset for the Social Media Engagement
Analyzer project. All data is fake, generated with a fixed random seed so
every run reproduces the same numbers. No external APIs are used.

The hidden "ground truth" rules baked into the generator (deliberately kept
here, not in the analysis script, so the analysis has to *discover* them):
  - Video/Reel posts earn the most engagement, then carousels, then images.
  - Engagement peaks around midday (12-14) and evening (19-22).
  - Weekends (Sat/Sun) perform a little better than weekdays.
  - 3-6 hashtags is the sweet spot; too many hurts.
  - Captions of ~80-220 characters do best; very long captions hurt.
  - Engagement rate drifts slightly down as the account grows.

Output: data/posts.csv  (~2,000 posts, Jan-Jun 2026)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

FORMATS = ["Reel", "Carousel", "Image", "Text"]
FORMAT_BOOST = {"Reel": 1.55, "Carousel": 1.25, "Image": 1.0, "Text": 0.70}
FORMAT_PROBS = [0.30, 0.30, 0.30, 0.10]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hour_effect(hour: int) -> float:
    """Engagement multiplier by hour of day (lunchtime + evening peaks)."""
    effect = 0.55
    if 12 <= hour <= 14:
        effect = 1.35
    elif 19 <= hour <= 22:
        effect = 1.45
    elif 7 <= hour <= 9:
        effect = 1.05
    elif 16 <= hour <= 18:
        effect = 1.10
    elif 0 <= hour <= 5:
        effect = 0.45
    return effect


def main() -> None:
    n_posts = 2000

    # Post dates spread over Jan-Jun 2026, a few posts per day.
    start = np.datetime64("2026-01-01")
    days_offset = RNG.integers(0, 181, size=n_posts)
    posted_at = start + days_offset.astype("timedelta64[D]")

    # Followers grow roughly linearly: 10k -> 40k over the period.
    followers = (10_000 + days_offset * 165 + RNG.normal(0, 800, n_posts)).astype(int)
    followers = followers.clip(min=5_000)

    post_format = RNG.choice(FORMATS, size=n_posts, p=FORMAT_PROBS)
    hour = RNG.integers(0, 24, size=n_posts)
    dow = pd.to_datetime(posted_at).dayofweek  # 0 = Monday
    caption_len = RNG.integers(20, 400, size=n_posts)
    hashtag_count = RNG.integers(0, 16, size=n_posts)
    mentions = RNG.integers(0, 4, size=n_posts)

    # --- ground-truth engagement model ---
    base_rate = 0.045
    fmt_mult = np.array([FORMAT_BOOST[f] for f in post_format])
    hour_mult = np.array([hour_effect(h) for h in hour])
    dow_mult = np.where(np.isin(dow, [5, 6]), 1.12, 1.0)
    hashtag_mult = np.where(
        (hashtag_count >= 3) & (hashtag_count <= 6), 1.12,
        np.where(hashtag_count > 10, 0.88, 1.0),
    )
    caption_mult = np.where(
        (caption_len >= 80) & (caption_len <= 220), 1.08,
        np.where(caption_len > 300, 0.90, 1.0),
    )
    growth_drag = 1 - (followers - followers.min()) / followers.max() * 0.25

    engagement_rate = (
        base_rate * fmt_mult * hour_mult * dow_mult
        * hashtag_mult * caption_mult * growth_drag
        * RNG.lognormal(0, 0.25, n_posts)
    ).clip(min=0.002, max=0.30)

    reach = (followers * RNG.uniform(0.25, 0.75, n_posts)).astype(int)
    total_engagements = (reach * engagement_rate).astype(int)
    # Typical split of engagements across likes/comments/shares.
    shares_p = RNG.uniform(0.06, 0.14, n_posts)
    comments_p = RNG.uniform(0.10, 0.20, n_posts)
    shares = (total_engagements * shares_p).astype(int)
    comments = (total_engagements * comments_p).astype(int)
    likes = (total_engagements - shares - comments).clip(min=0)

    df = pd.DataFrame({
        "post_id": [f"post_{i:04d}" for i in range(n_posts)],
        "posted_at": pd.to_datetime(posted_at),
        "hour": hour,
        "day_of_week": [DOW_NAMES[d] for d in dow],
        "format": post_format,
        "caption_length": caption_len,
        "hashtag_count": hashtag_count,
        "mentions": mentions,
        "followers_at_post": followers,
        "reach": reach,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "engagement_rate": (likes + comments + shares) / reach,
    })
    df["engagement_rate"] = df["engagement_rate"].round(4)

    df.to_csv("data/posts.csv", index=False)
    print(f"Wrote data/posts.csv with {len(df):,} posts")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
