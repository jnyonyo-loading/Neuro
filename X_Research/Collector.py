#!/usr/bin/env python3
"""
X Topic Agent — Collector
─────────────────────────
Searches X.com for posts, images, and links on a configured topic,
then saves results to data.json for the dashboard to display.

Run on a schedule (cron / Task Scheduler / GitHub Actions) to keep fresh.

Usage:
    python collector.py
    python collector.py --config path/to/config.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tweepy
except ImportError:
    sys.exit("tweepy not found — run: pip install -r requirements.txt")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("x-agent")

# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "config.json"
DEFAULT_OUTPUT = "data.json"


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}\nCopy config.json.example to config.json and fill in your details.")

    with open(cfg_path) as f:
        cfg = json.load(f)

    # allow env var override for bearer token (useful for CI/GitHub Actions)
    env_token = os.environ.get("X_BEARER_TOKEN") or os.environ.get("XTA_BEARER_TOKEN")
    if env_token:
        cfg["bearer_token"] = env_token
        log.info("Bearer token loaded from environment variable")

    required = {"bearer_token", "topic"}
    missing = required - set(cfg.keys())
    if missing:
        sys.exit(f"config.json is missing required keys: {missing}")

    if cfg["bearer_token"].startswith("YOUR_"):
        sys.exit("Please replace the placeholder bearer_token in config.json with your real X API Bearer Token.")

    return cfg


# ── api ───────────────────────────────────────────────────────────────────────

def build_client(cfg: dict) -> tweepy.Client:
    return tweepy.Client(bearer_token=cfg["bearer_token"], wait_on_rate_limit=True)


def build_query(cfg: dict) -> str:
    topic = cfg["topic"]
    parts = [topic]

    if cfg.get("exclude_retweets", True):
        parts.append("-is:retweet")
    if cfg.get("exclude_replies", True):
        parts.append("-is:reply")
    if lang := cfg.get("language"):
        parts.append(f"lang:{lang}")

    return " ".join(parts)


def fetch_posts(client: tweepy.Client, cfg: dict) -> list[dict]:
    query = build_query(cfg)
    max_results = min(int(cfg.get("max_results_per_run", 100)), 500)
    per_page = min(max_results, 100)
    pages = max(1, -(-max_results // per_page))  # ceiling division

    log.info("Query  : %s", query)
    log.info("Fetching up to %d posts (%d per page, %d pages)", max_results, per_page, pages)

    users_map: dict = {}
    media_map: dict = {}
    posts: list[dict] = []

    try:
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            max_results=per_page,
            limit=pages,
            tweet_fields=[
                "created_at", "author_id", "text", "public_metrics",
                "attachments", "entities", "lang",
            ],
            user_fields=["name", "username", "profile_image_url", "verified", "description", "public_metrics"],
            media_fields=["url", "preview_image_url", "type", "width", "height", "alt_text"],
            expansions=["author_id", "attachments.media_keys"],
        )

        for response in paginator:
            if not response.data:
                continue

            inc = response.includes or {}

            for u in inc.get("users", []):
                users_map[u.id] = {
                    "name":          u.name,
                    "username":      u.username,
                    "profile_image": getattr(u, "profile_image_url", None),
                    "verified":      bool(getattr(u, "verified", False)),
                    "bio":           getattr(u, "description", ""),
                    "followers":     (getattr(u, "public_metrics", None) or {}).get("followers_count", 0),
                }

            for m in inc.get("media", []):
                media_map[m.media_key] = {
                    "type":   m.type,
                    "url":    getattr(m, "url", None) or getattr(m, "preview_image_url", None),
                    "width":  getattr(m, "width",  None),
                    "height": getattr(m, "height", None),
                    "alt":    getattr(m, "alt_text", ""),
                }

            for tweet in response.data:
                posts.append(_parse_tweet(tweet, users_map, media_map))

    except tweepy.errors.Unauthorized:
        sys.exit("X API: Unauthorized — check your bearer_token in config.json.")
    except tweepy.errors.Forbidden as e:
        sys.exit(f"X API: Forbidden — your API plan may not support this endpoint.\n{e}")
    except tweepy.TweepyException as e:
        log.error("X API error: %s", e)

    log.info("Fetched %d posts from API", len(posts))
    return posts


def _parse_tweet(tweet, users_map: dict, media_map: dict) -> dict:
    author  = users_map.get(tweet.author_id, {})
    metrics = tweet.public_metrics or {}

    # media attachments
    media_items: list = []
    if tweet.attachments and tweet.attachments.get("media_keys"):
        for mk in tweet.attachments["media_keys"]:
            if mk in media_map:
                media_items.append(media_map[mk])

    # url cards + hashtags + mentions from entities
    urls:     list = []
    hashtags: list = []
    mentions: list = []

    if tweet.entities:
        for url_obj in tweet.entities.get("urls", []):
            expanded = url_obj.get("expanded_url", "")
            display  = url_obj.get("display_url", "")
            # skip t.co self-links and native X/Twitter media links
            if ("t.co" in display or "twitter.com" in expanded or "x.com/i/" in expanded):
                continue
            urls.append({
                "url":     expanded,
                "display": display,
                "title":   url_obj.get("title"),
                "desc":    url_obj.get("description"),
                "image":   (url_obj.get("images") or [{}])[0].get("url"),
            })

        for h in tweet.entities.get("hashtags", []):
            hashtags.append(h["tag"].lower())

        for m in tweet.entities.get("mentions", []):
            mentions.append(m["username"])

    username = author.get("username", "i")
    return {
        "id":         str(tweet.id),
        "text":       tweet.text,
        "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
        "lang":       getattr(tweet, "lang", None),
        "author":     author,
        "metrics": {
            "likes":       metrics.get("like_count",       0),
            "retweets":    metrics.get("retweet_count",    0),
            "replies":     metrics.get("reply_count",      0),
            "quotes":      metrics.get("quote_count",      0),
            "impressions": metrics.get("impression_count", 0),
        },
        "media":    media_items,
        "urls":     urls,
        "hashtags": hashtags,
        "mentions": mentions,
        "link":     f"https://x.com/{username}/status/{tweet.id}",
    }


# ── persistence ───────────────────────────────────────────────────────────────

def save(new_posts: list[dict], cfg: dict) -> dict:
    out_path   = Path(cfg.get("output_file", DEFAULT_OUTPUT))
    max_stored = int(cfg.get("max_stored", 500))

    # load existing archive
    existing: list[dict] = []
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f).get("posts", [])
        except (json.JSONDecodeError, KeyError):
            log.warning("Could not parse existing %s — starting fresh", out_path)

    # deduplicate by tweet ID
    known_ids  = {p["id"] for p in existing}
    new_unique = [p for p in new_posts if p["id"] not in known_ids]

    # merge, sort newest first, cap
    all_posts = new_unique + existing
    all_posts.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    all_posts = all_posts[:max_stored]

    output = {
        "topic":        cfg["topic"],
        "query":        build_query(cfg),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total":        len(all_posts),
        "new_this_run": len(new_unique),
        "posts":        all_posts,
    }

    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("Saved %d posts (%d new) → %s", len(all_posts), len(new_unique), out_path)
    return output


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="X Topic Agent — Collector")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.json")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("X Topic Agent starting")

    cfg    = load_config(args.config)
    log.info("Topic  : %s", cfg["topic"])

    client = build_client(cfg)
    posts  = fetch_posts(client, cfg)
    result = save(posts, cfg)

    log.info("Done   : %d total posts in store (%d new)", result["total"], result["new_this_run"])
    log.info("=" * 55)


if __name__ == "__main__":
    main()