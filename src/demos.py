"""Download and parse FACEIT demo files (experimental).

Bulk-downloads the demo recordings of a player's recent matches and
extracts per-round and per-player stats to JSON with demoparser2.
Not used by the current model - kept as groundwork for richer,
within-match features.

Requires the FACEIT_API_KEY environment variable to be set.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from demoparser2 import DemoParser

# ========= CONFIG =========
FACEIT_API_KEY = os.environ["FACEIT_API_KEY"]
PLAYER_NICKNAME = ""
MATCH_LIMIT = 1000    # how many recent matches to process
MAX_WORKERS = 5       # how many demos to process in parallel
OUTPUT_DIR = "parsed_demos"
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {FACEIT_API_KEY}"
}

# --------------------------
# Step 1: Get player ID from nickname
# --------------------------
def get_player_id(nickname):
    url = f"https://open.faceit.com/data/v4/players?nickname={nickname}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()["player_id"]

# --------------------------
# Step 2: Get match history
# --------------------------
def get_recent_matches(player_id, limit=1000):
    url = f"https://open.faceit.com/data/v4/players/{player_id}/history?game=cs2&limit={limit}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    items = r.json().get("items", [])
    return [match["match_id"] for match in items]

def get_demo_url(match_id):
    url = f"https://open.faceit.com/data/v4/matches/{match_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    match_data = r.json()
    
    demos = match_data.get("demo", [])
    if demos:
        # usually list of dicts with {"url": "..."}
        return demos[0]["url"]
    return None

# --------------------------
# Step 3: Download demo by match_id
# --------------------------
def download_demo(match_id, out_path):
    demo_url = get_demo_url(match_id)
    if not demo_url:
        print(f"[!] No demo URL found for {match_id}")
        return False

    print(f"[+] Downloading {match_id} from {demo_url[:80]}...")
    with requests.get(demo_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return True



# --------------------------
# Step 4: Parse demo
# --------------------------
def parse_demo(demo_path, out_json):
    parser = DemoParser(demo_path)
    data = {
        "match_id": os.path.basename(demo_path).replace(".dem", ""),
        "map": parser.match.map,
        "players": {},
        "rounds": []
    }

    for round_obj in parser.match.rounds:
        data["rounds"].append({
            "round_number": round_obj.number,
            "winner": round_obj.winner,
            "score_t": round_obj.score["T"],
            "score_ct": round_obj.score["CT"]
        })

    for player in parser.match.players:
        data["players"][player.steamid] = {
            "name": player.name,
            "team": player.team,
            "kills": player.kills,
            "deaths": player.deaths,
            "assists": player.assists,
            "hs": player.headshots,
            "side_start": player.starting_side
        }

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

# --------------------------
# Step 5: Process one match
# --------------------------
def process_match(match_id):
    demo_file = os.path.join(OUTPUT_DIR, f"{match_id}.dem")
    json_file = os.path.join(OUTPUT_DIR, f"{match_id}.json")

    if not download_demo(match_id, demo_file):
        return  # skip if download failed

    try:
        print(f"[+] Parsing {match_id}")
        parse_demo(demo_file, json_file)
    finally:
        if os.path.exists(demo_file):
            os.remove(demo_file)
        print(f"[+] Finished {match_id} -> {json_file if os.path.exists(json_file) else 'NO JSON'}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    player_id = get_player_id(PLAYER_NICKNAME)
    match_ids = get_recent_matches(player_id, MATCH_LIMIT)

    print(f"[+] Found {len(match_ids)} matches for {PLAYER_NICKNAME}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_match, mid) for mid in match_ids]
        for f in as_completed(futures):
            f.result()  # raise exceptions if any

    print("[+] All done!")
