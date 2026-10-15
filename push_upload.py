"""Push Day 8 (Social Media Engagement Analyzer) files to GitHub via push_files.
Usage: python3 push_upload.py text   -> pushes text files (one commit)
       python3 push_upload.py png     -> pushes 6 chart PNGs (one commit each)
"""
import json, subprocess, base64, os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
OWNER, REPO, BRANCH = "kashif-100", "daily-data-ai-2026-10-15", "main"

def push(files, message):
    args = {"owner": OWNER, "repo": REPO, "branch": BRANCH,
            "files": files, "message": message}
    payload = json.dumps(args)
    print(f"Pushing {len(files)} file(s), payload {len(payload)} bytes: {message[:60]}")
    result = subprocess.run(
        ["/opt/hatch/bin/github", "call-tool", "--name", "push_files",
         "--arguments-json", payload],
        capture_output=True, text=True, timeout=600)
    print("RC:", result.returncode)
    out = result.stdout.strip()
    print("STDOUT:", out[:1500])
    if result.stderr.strip():
        print("STDERR:", result.stderr.strip()[:800])
    return result.returncode == 0, out

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "text"
    ok_all = True
    if mode == "text":
        names = ["README.md", "requirements.txt", "generate_data.py",
                 "social_media_engagement.py", "ai_insights.py", "AI_INSIGHTS.md",
                 "INTERVIEW_NOTES.md", "findings.json", "push_upload.py"]
        files = []
        for n in names:
            with open(os.path.join(PROJ, n), encoding="utf-8") as fh:
                files.append({"path": n, "content": fh.read()})
        ok, _ = push(files, "Day 8: Social Media Engagement Analyzer - RandomForest, EDA, best-time analysis, AI insights")
        ok_all = ok
    elif mode == "csv":
        with open(os.path.join(PROJ, "data", "posts.csv"), encoding="utf-8") as fh:
            csv = fh.read()
        ok, _ = push([{"path": "data/posts.csv", "content": csv}],
                     "Day 8: add synthetic social media dataset (regenerate with generate_data.py)")
        ok_all = ok
    elif mode == "png":
        for n in ["actual_vs_predicted.png", "engagement_by_hour.png",
                  "engagement_by_dow.png", "engagement_by_format.png", "engagement_by_hashtags.png", "feature_importance.png"]:
            with open(os.path.join(PROJ, "charts", n), "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            ok, _ = push([{"path": f"charts/{n}", "content": b64}],
                         f"Day 8: add chart {n}")
            ok_all = ok_all and ok
            if not ok:
                print(f"STOPPING after failed push of {n}")
                break
    print("RESULT:", "OK" if ok_all else "FAILED")

if __name__ == "__main__":
    main()
