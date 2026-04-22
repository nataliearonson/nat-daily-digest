import os
import json
import socket
import feedparser
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Environment, FileSystemLoader
import anthropic

from sources import FEEDS

MAX_ENTRIES_PER_FEED = 5
FETCH_TIMEOUT = 10


def fetch_feed(feed):
    socket.setdefaulttimeout(FETCH_TIMEOUT)
    try:
        parsed = feedparser.parse(feed["url"])
        entries = []
        for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
            # Strip HTML tags from summary crudely
            import re
            summary = re.sub(r"<[^>]+>", "", summary).strip()
            summary = summary[:300] if summary else ""
            if title and link:
                entries.append({
                    "headline": title,
                    "link": link,
                    "summary": summary,
                    "source": feed["name"],
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
2. Select 10-15 interesting but lower-urgency stories for the WORTH READING LATER section.
3. Select 8-12 stories for the LESS IMPORTANT AND NOT URGENT section — these are topics a lot of people are talking about or that have cultural buzz, but that have little real-world impact or urgency. Think: celebrity news, viral moments, lighthearted trends, minor sports drama, pop culture. The reader just wants to be aware of what's in the conversation.
4. Apply a quality filter to sections 1 and 2 — exclude or demote stories that are:
   - Speculative or based on unnamed sources ("could", "might", "some say", "insiders claim")
   - Clickbait or emotionally manipulative
   - Health/science claims not backed by peer-reviewed research or expert consensus
   - Social media rumors dressed as news
5. Rank top headlines by newsworthiness and real-world impact, across all topics: business, markets, tech, Philly local, US politics, world events, sports, science/health, culture.

Return ONLY valid JSON in this exact format, no other text:
{
  "top_headlines": [
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


DAILY_QUOTES = {
    0: ("The only way to do great work is to love what you do.", "Steve Jobs"),
    1: ("The future belongs to those who believe in the beauty of their dreams.", "Eleanor Roosevelt"),
    2: ("Success is not final, failure is not fatal: it is the courage to continue that counts.", "Winston Churchill"),
    3: ("Believe you can and you're halfway there.", "Theodore Roosevelt"),
    4: ("The best way to predict the future is to create it.", "Peter Drucker"),
    5: ("Happiness is not something ready made. It comes from your own actions.", "Dalai Lama"),
    6: ("What you get by achieving your goals is not as important as what you become by achieving your goals.", "Zig Ziglar"),
}


def render_html(curated, source_count):
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.html")

    et = ZoneInfo("America/New_York")
    now = datetime.now(tz=et)
    date_str = now.strftime("%A, %B %-d, %Y")
    time_str = now.strftime("%-I:%M %p ET")
    quote_text, quote_author = DAILY_QUOTES[now.weekday()]

    html = template.render(
        date=date_str,
        updated_time=time_str,
        source_count=source_count,
        quote_text=quote_text,
        quote_author=quote_author,
        top_headlines=curated["top_headlines"],
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

    html = render_html(curated, source_count=len(FEEDS))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Written to index.html")


if __name__ == "__main__":
    main()
