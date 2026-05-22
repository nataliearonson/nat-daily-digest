import os
import json
import socket
import time as time_module
import feedparser
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Environment, FileSystemLoader
import anthropic

from sources import FEEDS

MAX_ENTRIES_PER_FEED = 5
FETCH_TIMEOUT = 10
MAX_AGE_DAYS = 5
RECENT_HOURS = 48


def parse_entry_date(entry):
    """Return the entry's published/updated time as a UTC datetime, or None."""
    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                return datetime.fromtimestamp(time_module.mktime(t), tz=timezone.utc)
            except Exception:
                pass
    return None


def fetch_feed(feed):
    socket.setdefaulttimeout(FETCH_TIMEOUT)
    try:
        parsed = feedparser.parse(feed["url"])
        entries = []
        now_utc = datetime.now(tz=timezone.utc)
        cutoff = now_utc - timedelta(days=MAX_AGE_DAYS)
        for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
            # Strip HTML tags from summary crudely
            import re
            summary = re.sub(r"<[^>]+>", "", summary).strip()
            summary = summary[:300] if summary else ""

            pub_dt = parse_entry_date(entry)
            # Drop anything older than MAX_AGE_DAYS
            if pub_dt and pub_dt < cutoff:
                continue

            age_label = ""
            if pub_dt:
                age_hours = (now_utc - pub_dt).total_seconds() / 3600
                if age_hours > RECENT_HOURS:
                    days = int(age_hours // 24)
                    age_label = f"{days} days ago"

            if title and link:
                entries.append({
                    "headline": title,
                    "link": link,
                    "summary": summary,
                    "source": feed["name"],
                    "age_label": age_label,
                })
        return entries
    except Exception as e:
        print(f"  [skip] {feed['name']}: {e}")
        return []


def fetch_all_feeds():
    all_entries = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_feed, feed): feed for feed in FEEDS}
        for future in as_completed(futures):
            entries = future.result()
            all_entries.extend(entries)
            if entries:
                print(f"  [ok] {futures[future]['name']}: {len(entries)} entries")
    return all_entries


def deduplicate(entries):
    seen = set()
    unique = []
    for entry in entries:
        key = entry["headline"].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def curate_with_claude(entries):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    headlines_text = "\n".join(
        f"[{i+1}] SOURCE: {e['source']}\n    HEADLINE: {e['headline']}\n    SUMMARY: {e['summary']}\n    LINK: {e['link']}"
        for i, e in enumerate(entries)
    )

    system_prompt = """You are a sharp, experienced news editor curating a daily digest for a single reader.

Your job:
1. Select the 15-20 most important, newsworthy headlines for the TOP HEADLINES section.
2. Select 5-10 Philadelphia or Pennsylvania-specific headlines for the PHILADELPHIA & PA section. These can be any topic (politics, local news, business, sports, culture, etc.) as long as they are PA/Philly focused. Do NOT duplicate items already chosen for TOP HEADLINES.
3. Select 5-10 education or technology headlines for the EDUCATION & TECHNOLOGY section. Include US and international stories about schools, universities, edtech, AI, software, hardware, internet, cybersecurity, scientific computing, etc. These MAY overlap with items in other sections — the overlap will be labeled automatically, so pick the best edu/tech items even if they appear elsewhere.
4. Select 10-15 interesting but lower-urgency stories for the WORTH READING LATER section.
5. Select 8-12 stories for the LESS IMPORTANT AND NOT URGENT section — these are topics a lot of people are talking about or that have cultural buzz, but that have little real-world impact or urgency. Think: celebrity news, viral moments, lighthearted trends, minor sports drama, pop culture. The reader just wants to be aware of what's in the conversation.
6. Apply a quality filter to sections 1, 2, 3, and 4 — exclude or demote stories that are:
   - Speculative or based on unnamed sources ("could", "might", "some say", "insiders claim")
   - Clickbait or emotionally manipulative
   - Health/science claims not backed by peer-reviewed research or expert consensus
   - Social media rumors dressed as news
7. Rank top headlines by newsworthiness and real-world impact, across all topics: business, markets, tech, Philly local, US politics, world events, sports, science/health, culture.

Return ONLY valid JSON in this exact format, no other text:
{
  "top_headlines": [
    {"headline": "...", "source": "...", "link": "...", "summary": "one sentence, plain text"}
  ],
  "philly_pa": [
    {"headline": "...", "source": "...", "link": "...", "summary": "one sentence, plain text"}
  ],
  "education_tech": [
    {"headline": "...", "source": "...", "link": "...", "summary": "one sentence, plain text"}
  ],
  "worth_reading_later": [
    {"headline": "...", "source": "...", "link": "...", "summary": ""}
  ],
  "less_important": [
    {"headline": "...", "source": "...", "link": "...", "summary": ""}
  ]
}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Here are today's headlines from trusted news sources. Curate the digest:\n\n{headlines_text}",
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    # Extract JSON object robustly
    start = raw.find("{")
    end = raw.rfind("}") + 1
    raw = raw[start:end]

    return json.loads(raw)


def render_html(curated, source_count):
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.html")

    et = ZoneInfo("America/New_York")
    now = datetime.now(tz=et)
    date_str = now.strftime("%A, %B %-d, %Y")
    time_str = now.strftime("%-I:%M %p ET")

    html = template.render(
        date=date_str,
        updated_time=time_str,
        source_count=source_count,
        top_headlines=curated["top_headlines"],
        philly_pa=curated.get("philly_pa", []),
        education_tech=curated.get("education_tech", []),
        worth_reading_later=curated["worth_reading_later"],
        less_important=curated.get("less_important", []),
    )
    return html


def main():
    print("Fetching feeds...")
    entries = fetch_all_feeds()
    print(f"Fetched {len(entries)} raw entries")

    entries = deduplicate(entries)
    print(f"{len(entries)} entries after deduplication")

    print("Curating with Claude...")
    curated = curate_with_claude(entries)
    print(f"Top headlines: {len(curated['top_headlines'])}, Worth reading later: {len(curated['worth_reading_later'])}")

    # Re-attach age_label from original entries (Claude doesn't see it)
    age_map = {e["link"]: e.get("age_label", "") for e in entries}
    all_sections = ("top_headlines", "philly_pa", "education_tech", "worth_reading_later", "less_important")
    for section in all_sections:
        for item in curated.get(section, []):
            item["age_label"] = age_map.get(item.get("link", ""), "")

    # Label Education & Technology items that also appear in another section
    section_display_names = {
        "top_headlines": "Top Headlines",
        "philly_pa": "Philadelphia & PA",
        "worth_reading_later": "Worth Reading Later",
        "less_important": "Less Important",
    }
    link_to_section = {}
    for key, label in section_display_names.items():
        for item in curated.get(key, []):
            link = item.get("link", "")
            if link and link not in link_to_section:
                link_to_section[link] = label
    for item in curated.get("education_tech", []):
        item["overlap_label"] = link_to_section.get(item.get("link", ""), "")

    html = render_html(curated, source_count=len(FEEDS))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Written to index.html")


if __name__ == "__main__":
    main()
