#!/usr/bin/env python3
# wikipedia_infinite_crawler.py
# Python 3


import requests
import re
import time
import traceback
from collections import deque, Counter

WIKI_API = "https://en.wikipedia.org/w/api.php"
OUTPUT_FILE = "data.txt"
USER_AGENT = "WikiInfiniteCrawler/1.0 (+https://example.com)"
REQUEST_DELAY = 1.0   # polite delay between requests (seconds)
TOP_N_RESULTS = 3
DERIVED_PER_DESC = 3  # how many words to derive from each description

# initial seed keywords (you can change)
SEED_KEYWORDS = [
"technology",
"artificial intelligence",
"machine learning",
"deep learning",
"natural language processing",
"computer vision",
"robotics",
"quantum computing",
"data science",
"big data",
"cloud computing",
"distributed systems",
"blockchain",
"cryptocurrencies",
"fintech",
"cybersecurity",
"information security",
"edge computing",
"internet of things",
"5g networks",
"wireless networking",
"semiconductors",
"nanotechnology",
"materials science",
"metamaterials",
"3d printing",
"additive manufacturing",
"renewable energy",
"solar energy",
"wind energy",
"battery technology",
"energy storage",
"electric vehicles",
"autonomous vehicles",
"aerospace engineering",
"space exploration",
"astrophysics",
"cosmology",
"particle physics",
"quantum mechanics",
"condensed matter physics",
"nuclear physics",
"computational physics",
"applied mathematics",
"statistics",
"probability theory",
"numerical analysis",
"optimization",
"operations research",
"control theory",
"signal processing",
"computer graphics",
"human-computer interaction",
"user experience",
"software engineering",
"programming languages",
"functional programming",
"devops",
"containerization",
"kubernetes",
"web development",
"frontend development",
"backend development",
"apis",
"databases",
"sql",
"nosql",
"data engineering",
"data visualization",
"bioinformatics",
"computational biology",
"genomics",
"crispr",
"synthetic biology",
"molecular biology",
"cell biology",
"microbiology",
"immunology",
"virology",
"epidemiology",
"public health",
"precision medicine",
"oncology",
"neuroscience",
"cognitive science",
"psychology",
"behavioral economics",
"economics",
"macroeconomics",
"microeconomics",
"finance",
"financial engineering",
"risk management",
"supply chain management",
"logistics",
"operations management",
"entrepreneurship",
"startups",
"product management",
"marketing",
"digital marketing",
"social media",
"content strategy",
"game design",
"virtual reality",
"augmented reality",
"mixed reality",
"metaverse",
"human rights",
"political science",
"international relations",
"law",
"intellectual property",
"ethics",
"ai ethics",
"philosophy",
"history",
"archaeology",
"linguistics",
"translation studies",
"education technology",
"pedagogy",
"healthcare technology",
"medical devices",
"telemedicine",
"wearable technology",
"biotechnology",
"proteomics",
"metabolomics",
"systems biology",
"climate change",
"environmental science",
"ecology",
"conservation biology",
"oceanography",
"geology",
"geophysics",
"urban planning",
"architecture",
"civil engineering",
"structural engineering",
"chemical engineering",
"automotive engineering",
"manufacturing",
"industrial design",
"human factors",
"ergonomics",
"climate policy",
"renewable materials",
"food science",
"agriculture technology",
"precision agriculture",
"quantum chemistry",
"computational chemistry",
"catalysis",
"photonics",
"optics",
"sensor technology",
"remote sensing",
"geographic information systems"
]


# small stopword set (English) to avoid useless derived words
STOPWORDS = {
    "the","and","for","that","with","this","from","have","are","was","were","will","has",
    "had","but","not","you","your","they","their","its","can","all","one","about","which",
    "when","what","how","why","where","also","other","such","these","those","each","may",
    "into","over","more","most","some","any","use","used","using","than","then","there","here",
    "our","we","i","he","she","it","on","in","is","a","an","to","of","by","be"
}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

def wiki_search_titles(query, limit=TOP_N_RESULTS):
    """
    Use MediaWiki 'search' API to get top page titles for query.
    Returns list of titles (strings).
    """
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1
        }
        r = session.get(WIKI_API, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        hits = j.get("query", {}).get("search", [])
        titles = [h.get("title") for h in hits if h.get("title")]
        return titles
    except Exception as e:
        print(f"[WARN] wiki_search_titles failed for '{query}': {e}")
        return []

def wiki_get_intro_by_titles(titles):
    """
    Given a list of page titles, fetch plain-text intro extracts.
    Returns dict title -> intro_text (may be empty string).
    """
    if not titles:
        return {}
    try:
        # join titles by | for API
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": "|".join(titles),
            "format": "json",
            "utf8": 1
        }
        r = session.get(WIKI_API, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        out = {}
        for p in pages.values():
            title = p.get("title", "")
            extract = p.get("extract", "") or ""
            out[title] = extract
        return out
    except Exception as e:
        print(f"[WARN] wiki_get_intro_by_titles failed: {e}")
        return {t: "" for t in titles}

def sanitize_single_line(text):
    """Remove newlines and compress whitespace to single spaces; strip."""
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    # ensure no newline characters remain
    s = s.replace("\n", " ").replace("\r", " ")
    return s

def extract_candidate_words(text, top_k=DERIVED_PER_DESC):
    """
    Extract candidate words from text: alphabetic words of length>=4,
    lowercased, filtered by STOPWORDS, return top_k most common.
    """
    if not text:
        return []
    words = re.findall(r"[A-Za-z]{4,}", text)
    words = [w.lower() for w in words]
    words = [w for w in words if w not in STOPWORDS]
    if not words:
        return []
    counts = Counter(words)
    most = [w for w,_ in counts.most_common(top_k)]
    return most

def append_line_to_file(line):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

def main_loop(seed_keywords):
    q = deque(seed_keywords)
    processed_count = 0
    print(f"[INFO] Starting infinite Wikipedia crawler. Output file: {OUTPUT_FILE}")
    print("[INFO] Stop with Ctrl+C")
    try:
        while True:
            if not q:
                # If queue empties, re-seed to keep infinite (you can change behavior)
                q.extend(seed_keywords)
            query = q.popleft()
            query = sanitize_single_line(query)
            if not query:
                continue
            print(f"[INFO] Searching Wikipedia for: '{query}'")
            titles = wiki_search_titles(query, limit=TOP_N_RESULTS)
            if not titles:
                print(f"[WARN] No wiki results for '{query}'")
                time.sleep(REQUEST_DELAY)
                continue
            intros = wiki_get_intro_by_titles(titles)
            for title in titles:
                intro = intros.get(title, "")
                single_intro = sanitize_single_line(intro)
                # WRITE: exactly "Title Description" on one line (no pipes)
                output_line = f"{title} {single_intro}"
                append_line_to_file(output_line)
                processed_count += 1
                print(f"[LOG] Wrote ({processed_count}) -> {title}")
                # derive new keywords from description
                derived = extract_candidate_words(single_intro, top_k=DERIVED_PER_DESC)
                # enqueue derived words (append to right) to continue infinitely
                for d in derived:
                    q.append(d)
            # polite delay
            time.sleep(REQUEST_DELAY)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"[ERROR] Unexpected exception in main_loop: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # ensure output file exists (append mode will create)
    open(OUTPUT_FILE, "a", encoding="utf-8").close()
    main_loop(SEED_KEYWORDS)
