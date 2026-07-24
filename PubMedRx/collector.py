#!/usr/bin/env python3
"""
PubMed Research Agent — Collector
──────────────────────────────────
Searches PubMed for recent papers across all configured topics,
saves results to data.json for the dashboard to display.

Usage:
    python3 collector.py
    python3 collector.py --config path/to/config.json
"""

import argparse
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not found — run: pip3 install requests")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pubmed-agent")

# ── constants ─────────────────────────────────────────────────────────────────
BASE_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SLOW_DELAY = 0.4   # ~3 req/sec — NCBI limit without API key
FAST_DELAY = 0.12  # ~10 req/sec — NCBI limit with free API key


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not cfg.get("email"):
        sys.exit("config.json needs an 'email' field — required by NCBI.")
    if not cfg.get("topics"):
        sys.exit("config.json needs a 'topics' list with at least one entry.")
    return cfg


# ── api helpers ───────────────────────────────────────────────────────────────

def base_params(cfg: dict) -> dict:
    p = {
        "email": cfg["email"],
        "tool":  "pubmed-research-agent",
        "db":    "pubmed",
    }
    if cfg.get("api_key"):
        p["api_key"] = cfg["api_key"]
    return p


def search_ids(topic: str, cfg: dict) -> list[str]:
    """Run esearch — returns list of PMIDs for this topic."""
    params = base_params(cfg)
    params.update({
        "term":     topic,
        "retmax":   int(cfg.get("max_results_per_topic", 10)),
        "datetype": "pdat",
        "reldate":  int(cfg.get("days_back", 90)),
        "retmode":  "json",
        "sort":     "relevance",
    })
    r = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch_records(pmids: list[str], cfg: dict) -> list[dict]:
    """Run efetch for a batch of PMIDs — returns parsed paper dicts."""
    if not pmids:
        return []
    params = base_params(cfg)
    params.update({
        "id":      ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    })
    r = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return _parse_xml(r.text)


# ── xml parsing ───────────────────────────────────────────────────────────────

def _parse_xml(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.error("XML parse error: %s", e)
        return []
    papers = []
    for article in root.findall(".//PubmedArticle"):
        try:
            papers.append(_parse_article(article))
        except Exception as e:
            log.warning("Skipped article: %s", e)
    return papers


def _parse_article(article) -> dict:
    mc  = article.find("MedlineCitation")
    art = mc.find("Article")

    # PMID
    pmid = (mc.findtext("PMID") or "").strip()

    # Title — itertext handles italic/bold sub-elements
    title_el = art.find("ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    # Abstract — handles structured abstracts with labels
    abstract = ""
    abs_el = art.find("Abstract")
    if abs_el is not None:
        parts = []
        for t in abs_el.findall("AbstractText"):
            label = t.get("Label")
            text  = "".join(t.itertext()).strip()
            parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts)

    # Authors
    authors = []
    for author in art.findall(".//Author"):
        last  = author.findtext("LastName")  or ""
        first = author.findtext("ForeName")  or author.findtext("Initials") or ""
        name  = f"{last} {first}".strip()
        if name:
            authors.append(name)

    # Journal
    journal = (
        art.findtext(".//Journal/Title") or
        art.findtext(".//Journal/ISOAbbreviation") or ""
    ).strip()

    # Publication date
    date_el = art.find(".//Journal/JournalIssue/PubDate")
    pub_date = ""
    if date_el is not None:
        year  = date_el.findtext("Year")       or ""
        month = date_el.findtext("Month")      or ""
        day   = date_el.findtext("Day")        or ""
        mdate = date_el.findtext("MedlineDate") or ""
        pub_date = mdate if mdate else " ".join(filter(None, [year, month, day]))

    # DOI
    doi = ""
    for aid in article.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = (aid.text or "").strip()
            break

    # Keywords + MeSH
    keywords = [k.text.strip() for k in mc.findall(".//Keyword") if k.text]
    mesh     = [
        m.findtext("DescriptorName", "").strip()
        for m in mc.findall(".//MeshHeading")
        if m.findtext("DescriptorName")
    ]

    return {
        "pmid":     pmid,
        "title":    title,
        "abstract": abstract,
        "authors":  authors,
        "journal":  journal,
        "pub_date": pub_date,
        "doi":      doi,
        "keywords": keywords[:10],
        "mesh":     mesh[:8],
        "link":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


# ── collection loop ───────────────────────────────────────────────────────────

def collect(cfg: dict) -> list[dict]:
    topics = cfg["topics"]
    delay  = FAST_DELAY if cfg.get("api_key") else SLOW_DELAY
    all_papers: list[dict] = []

    for i, topic in enumerate(topics, 1):
        log.info("[%d/%d] %s", i, len(topics), topic)
        try:
            pmids = search_ids(topic, cfg)
            log.info("       Found %d paper IDs", len(pmids))
            time.sleep(delay)

            if pmids:
                papers = fetch_records(pmids, cfg)
                for p in papers:
                    p["topic"] = topic
                all_papers.extend(papers)
                log.info("       Fetched %d records", len(papers))

            time.sleep(delay)
        except requests.HTTPError as e:
            log.error("       HTTP error: %s", e)
        except Exception as e:
            log.error("       Error: %s", e)

    return all_papers


# ── persistence ───────────────────────────────────────────────────────────────

def save(new_papers: list[dict], cfg: dict) -> dict:
    out_path   = Path(cfg.get("output_file", "data.json"))
    max_stored = int(cfg.get("max_stored", 1000))

    existing: list[dict] = []
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f).get("papers", [])
        except (json.JSONDecodeError, KeyError):
            log.warning("Could not parse existing %s — starting fresh", out_path)

    known_ids  = {p["pmid"] for p in existing}
    new_unique = [p for p in new_papers if p["pmid"] not in known_ids]

    all_papers = new_unique + existing
    all_papers = all_papers[:max_stored]

    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total":        len(all_papers),
        "new_this_run": len(new_unique),
        "topics":       cfg["topics"],
        "papers":       all_papers,
    }

    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("Saved %d papers (%d new) → %s", len(all_papers), len(new_unique), out_path)
    return output


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PubMed Research Agent — Collector")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("PubMed Research Agent starting")

    cfg    = load_config(args.config)
    log.info("%d topics configured", len(cfg["topics"]))

    papers = collect(cfg)
    result = save(papers, cfg)

    log.info("Done — %d total papers (%d new)", result["total"], result["new_this_run"])
    log.info("=" * 55)


if __name__ == "__main__":
    main()