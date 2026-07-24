# X Topic Agent

Automatically compile posts, images, and links from X.com on any topic — and view them in a live dashboard.

```
collector.py  →  data.json  →  dashboard.html
(runs on schedule)   (data store)   (live dashboard)
```

---

## 1 · Get Your X API Bearer Token

1. Go to [developer.x.com](https://developer.x.com) and sign in
2. Create a new project + app (the free Basic plan works)
3. Navigate to your app → **Keys and Tokens**
4. Copy the **Bearer Token**

> The free Basic plan ($0/month) allows up to 10,000 tweet reads/month and is enough for hourly collection.

---

## 2 · Configure

Open `config.json` and fill in your details:

```json
{
  "bearer_token": "AAAAAAAAAAAAAAAAAAAAAxxxxxxxxxx",

  "topic": "#AI OR \"artificial intelligence\"",

  "language": "en",
  "exclude_retweets": true,
  "exclude_replies": true,
  "max_results_per_run": 100,
  "max_stored": 500,
  "output_file": "data.json"
}
```

**Topic syntax examples:**

| Goal | Query |
|------|-------|
| Hashtag | `#ClimateChange` |
| Phrase | `"machine learning"` |
| Either/or | `#AI OR "artificial intelligence"` |
| Specific account | `from:OpenAI` |
| Keyword + hashtag | `climate #COP30` |
| Keyword exclusion | `"Python" -snake` |

Full syntax: [developer.x.com/en/docs/twitter-api/tweets/search/integrate/build-a-query](https://developer.x.com/en/docs/twitter-api/tweets/search/integrate/build-a-query)

---

## 3 · Install & Run

```bash
# Install dependency
pip install -r requirements.txt

# Run the collector (generates data.json)
python collector.py
```

You should see output like:
```
2025-01-15 09:00:00  INFO  X Topic Agent starting
2025-01-15 09:00:00  INFO  Topic  : #AI OR "artificial intelligence"
2025-01-15 09:00:00  INFO  Query  : #AI OR "artificial intelligence" -is:retweet -is:reply lang:en
2025-01-15 09:00:01  INFO  Fetched 87 posts from API
2025-01-15 09:00:01  INFO  Saved 87 posts (87 new) → data.json
2025-01-15 09:00:01  INFO  Done   : 87 total posts in store (87 new)
```

---

## 4 · Open the Dashboard

The dashboard reads `data.json` via `fetch()`, so it needs a local HTTP server:

```bash
# In the x-topic-agent folder:
python -m http.server 8080
```

Then open: **http://localhost:8080/dashboard.html**

The dashboard auto-refreshes every 5 minutes and shows:
- Total posts / new this run / peak likes
- Top hashtags
- Filter by: All / Images / Links / Text only
- Sort by: Newest / Most liked / Most retweeted / Most engaged
- Search posts, @handles, or #tags
- Full engagement metrics per post

---

## 5 · Schedule the Collector

Pick the method that fits your environment:

### Mac / Linux (cron)

```bash
# Open your crontab
crontab -e

# Add one of these lines:

# Every hour
0 * * * * cd /path/to/x-topic-agent && python3 collector.py >> logs/collector.log 2>&1

# Every 30 minutes
*/30 * * * * cd /path/to/x-topic-agent && python3 collector.py >> logs/collector.log 2>&1

# Daily at 8am
0 8 * * * cd /path/to/x-topic-agent && python3 collector.py >> logs/collector.log 2>&1
```

> Find your Python path with: `which python3`

---

### Windows (Task Scheduler)

Run in PowerShell (as Administrator):

```powershell
$action = New-ScheduledTaskAction `
  -Execute "python" `
  -Argument "C:\path\to\x-topic-agent\collector.py" `
  -WorkingDirectory "C:\path\to\x-topic-agent"

$trigger = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 1) -Once -At (Get-Date)

Register-ScheduledTask -TaskName "X Topic Collector" -Action $action -Trigger $trigger
```

---

### Cloud (GitHub Actions) — Dashboard on GitHub Pages

This is the cleanest option: Actions runs the collector on a schedule, commits `data.json` back to the repo, and GitHub Pages hosts the dashboard publicly.

1. Push this folder to a GitHub repo
2. Add your Bearer Token as a repository secret named `X_BEARER_TOKEN`
3. Create `.github/workflows/collect.yml`:

```yaml
name: Collect X Data

on:
  schedule:
    - cron: '0 * * * *'   # every hour (UTC)
  workflow_dispatch:        # allow manual trigger

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: pip install tweepy

      - name: Run collector
        run: python collector.py
        env:
          X_BEARER_TOKEN: ${{ secrets.X_BEARER_TOKEN }}

      - name: Commit updated data.json
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: update data.json"
          file_pattern: data.json
```

4. Enable **GitHub Pages** (Settings → Pages → Deploy from branch `main`)
5. Your dashboard lives at: `https://YOUR_USERNAME.github.io/REPO_NAME/dashboard.html`

---

## Tips

**Rate limits:** The Basic plan allows 10,000 tweet reads/month. Running every hour at 100 results = ~72,000/month — upgrade to the Pro plan ($5,000/month) or reduce frequency / max_results if needed.

**Multiple topics:** Duplicate the folder and point each config to a different `output_file`.

**Keep logs:** The `logs/` folder is created automatically. Rotate logs with `logrotate` on Linux or periodically delete old ones.

**Tune max_stored:** Set higher (e.g. 2000) to keep a longer archive, or lower to keep the dashboard snappy.
