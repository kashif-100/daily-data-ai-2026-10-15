"""
social_media_engagement.py
--------------------------
Main analysis script. Runs end-to-end:

  1. Loads (or generates) data/posts.csv
  2. EDA: engagement by hour, day of week, format, hashtag count
  3. ML: RandomForestRegressor predicts engagement_rate; feature importances
  4. Saves 6 charts into charts/ and a machine-readable findings.json

Run:  python social_media_engagement.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FORMAT_ORDER = ["Reel", "Carousel", "Image", "Text"]
CHARTS_DIR = "charts"


def engagement_by_hour(df: pd.DataFrame) -> pd.Series:
    by_hour = df.groupby("hour")["engagement_rate"].mean()
    plt.figure(figsize=(10, 4.5))
    by_hour.plot(kind="bar", color="steelblue")
    plt.title("Average Engagement Rate by Hour of Day")
    plt.xlabel("Hour posted (24h)")
    plt.ylabel("Avg engagement rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/engagement_by_hour.png", dpi=120)
    plt.close()
    return by_hour


def engagement_by_dow(df: pd.DataFrame) -> pd.Series:
    by_dow = df.groupby("day_of_week")["engagement_rate"].mean().reindex(DOW_ORDER)
    plt.figure(figsize=(8, 4.5))
    by_dow.plot(kind="bar", color="teal")
    plt.title("Average Engagement Rate by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Avg engagement rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/engagement_by_dow.png", dpi=120)
    plt.close()
    return by_dow


def engagement_by_format(df: pd.DataFrame) -> pd.Series:
    by_format = (
        df.groupby("format")["engagement_rate"].mean().reindex(FORMAT_ORDER)
    )
    plt.figure(figsize=(7, 4.5))
    by_format.plot(kind="bar", color="coral")
    plt.title("Average Engagement Rate by Content Format")
    plt.xlabel("Format")
    plt.ylabel("Avg engagement rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/engagement_by_format.png", dpi=120)
    plt.close()
    return by_format


def engagement_by_hashtags(df: pd.DataFrame) -> pd.Series:
    # Bucket hashtag counts so the chart is readable.
    df = df.copy()
    df["hashtag_bucket"] = pd.cut(
        df["hashtag_count"],
        bins=[-1, 2, 6, 10, 20],
        labels=["0-2", "3-6", "7-10", "11+"],
    )
    by_tag = df.groupby("hashtag_bucket", observed=True)["engagement_rate"].mean()
    plt.figure(figsize=(7, 4.5))
    by_tag.plot(kind="bar", color="mediumpurple")
    plt.title("Average Engagement Rate by Hashtag Count")
    plt.xlabel("Hashtags used")
    plt.ylabel("Avg engagement rate")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/engagement_by_hashtags.png", dpi=120)
    plt.close()
    return by_tag


def build_model(df: pd.DataFrame):
    """Predict engagement_rate from post features with a Random Forest."""
    features = pd.get_dummies(
        df[["hour", "day_of_week", "format", "caption_length",
            "hashtag_count", "mentions", "followers_at_post"]],
        columns=["day_of_week", "format"],
    )
    target = df["engagement_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Model test MAE: {mae:.4f} | R^2: {r2:.3f}")

    # Actual vs predicted scatter.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.3, s=12)
    lims = [0, max(y_test.max(), preds.max()) * 1.05]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("Actual engagement rate")
    plt.ylabel("Predicted engagement rate")
    plt.title(f"Actual vs Predicted Engagement (R² = {r2:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/actual_vs_predicted.png", dpi=120)
    plt.close()

    # Feature importances.
    importances = (
        pd.Series(model.feature_importances_, index=features.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", color="darkgreen")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/feature_importance.png", dpi=120)
    plt.close()

    return model, features.columns.tolist(), mae, r2, importances


def main() -> None:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if not os.path.exists("data/posts.csv"):
        print("data/posts.csv not found — generating it first...")
        import generate_data
        generate_data.main()

    df = pd.read_csv("data/posts.csv", parse_dates=["posted_at"])
    print(f"Loaded {len(df):,} posts "
          f"({df['posted_at'].min():%Y-%m-%d} to {df['posted_at'].max():%Y-%m-%d})")
    print(f"Overall average engagement rate: {df['engagement_rate'].mean():.4f}")

    by_hour = engagement_by_hour(df)
    by_dow = engagement_by_dow(df)
    by_format = engagement_by_format(df)
    by_tag = engagement_by_hashtags(df)
    model, feature_names, mae, r2, importances = build_model(df)

    # "Best combination" recommendation: highest predicted engagement cell.
    grid = df.groupby(["format", "day_of_week"])["engagement_rate"].mean()
    best_combo = grid.idxmax()

    findings = {
        "n_posts": int(len(df)),
        "overall_avg_engagement_rate": round(float(df["engagement_rate"].mean()), 4),
        "top_3_hours": [int(h) for h in by_hour.sort_values(ascending=False).head(3).index],
        "best_day": str(by_dow.idxmax()),
        "worst_day": str(by_dow.idxmin()),
        "best_format": str(by_format.idxmax()),
        "worst_format": str(by_format.idxmin()),
        "format_avg_rates": {k: round(float(v), 4) for k, v in by_format.items()},
        "best_hashtag_bucket": str(by_tag.idxmax()),
        "avg_caption_length": int(df["caption_length"].mean()),
        "best_format_day_combo": {"format": best_combo[0], "day": best_combo[1]},
        "model": {
            "name": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 10,
            "test_mae": round(float(mae), 4),
            "test_r2": round(float(r2), 3),
            "features": feature_names,
        },
        "top_features": [str(f) for f in importances.head(5).index],
    }
    with open("findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    print("Best posting hours:", findings["top_3_hours"])
    print("Best format:", findings["best_format"],
          "| Best day:", findings["best_day"])
    print("Saved findings.json and 6 charts. Run ai_insights.py next.")


if __name__ == "__main__":
    main()
