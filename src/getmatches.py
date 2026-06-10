"""Helpers for the FACEIT Data API (v4).

Every function takes a `headers` dict containing the API authorization,
built once in the calling script:

    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
"""

from datetime import datetime, timezone

import pytz
import requests

BASE_URL = "https://open.faceit.com/data/v4"
GAME = "cs2"


def get_player_id(player_name, headers):
    """Return the FACEIT player id for a nickname, or None if not found."""
    response = requests.get(
        f"{BASE_URL}/players?nickname={player_name}&game={GAME}", headers=headers
    )
    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

    player = response.json()
    if "player_id" not in player:
        print("Player not found")
        return None
    return player["player_id"]


def get_elo(player_name, headers):
    """Return the player's current FACEIT elo (0 on request failure)."""
    response = requests.get(
        f"{BASE_URL}/players?nickname={player_name}&game={GAME}", headers=headers
    )
    if response.status_code != 200:
        return 0
    return response.json().get("games", {}).get(GAME, {}).get("faceit_elo", "N/A")


def get_player_matches(player_name, headers, num_matches):
    """Return up to `num_matches` recent match ids for a player.

    The history endpoint pages 100 matches at a time, so multiple
    requests are made when more than 100 matches are needed.
    """
    player_id = get_player_id(player_name, headers)
    if player_id is None:
        exit()

    pages = -(-num_matches // 100)  # the history endpoint returns at most 100 per page
    matches_id = []

    for page in range(pages):
        limit = min(100, num_matches - page * 100)
        url = (
            f"{BASE_URL}/players/{player_id}/history"
            f"?game={GAME}&limit={limit}&offset={page * 100}"
        )
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve match data: {response.status_code}")
            exit()

        for match in response.json().get("items", []):
            match_id = match.get("match_id", "N/A")
            if match_id not in matches_id:
                matches_id.append(match_id)

    return matches_id


def _get_player_stat(player_name, match_id, headers, stat, cast):
    """Return a single stat ('Kills', 'Deaths', 'ADR', ...) of a player in a match."""
    response = requests.get(f"{BASE_URL}/matches/{match_id}/stats", headers=headers)
    value = cast(0)

    for round_ in response.json().get("rounds", []):
        for team in round_["teams"]:
            for player in team["players"]:
                if player["nickname"] == player_name:
                    value = cast(player["player_stats"][stat])
    return value


def get_player_kills(player_name, match_id, headers):
    """Return the player's kill count in a match."""
    return _get_player_stat(player_name, match_id, headers, "Kills", int)


def get_player_deaths(player_name, match_id, headers):
    """Return the player's death count in a match."""
    return _get_player_stat(player_name, match_id, headers, "Deaths", int)


def get_player_damage(player_name, match_id, headers):
    """Return the player's average damage per round (ADR) in a match."""
    return _get_player_stat(player_name, match_id, headers, "ADR", float)


def get_enemy_team_elos(player_name, match_id, headers):
    """Return the average elo rating of the team the player faced (0 if unknown)."""
    response = requests.get(f"{BASE_URL}/matches/{match_id}", headers=headers)
    teams = response.json().get("teams", {})

    for faction in teams.values():
        roster = faction.get("roster", [])
        if all(player["nickname"] != player_name for player in roster):
            return faction.get("stats", {}).get("rating", 0)
    return 0


def get_match_time(match_id, headers):
    """Return the match start time as seconds since midnight (Europe/Helsinki).

    Only the time of day matters for the model, not the date.
    """
    response = requests.get(f"{BASE_URL}/matches/{match_id}", headers=headers)
    if response.status_code != 200:
        return 0

    started_at = int(response.json().get("started_at", "0"))
    match_time = datetime.fromtimestamp(started_at, tz=timezone.utc)
    match_time = match_time.astimezone(pytz.timezone("Europe/Helsinki"))
    return match_time.hour * 3600 + match_time.minute * 60 + match_time.second


def get_match_map(match_id, headers):
    """Return the map the match was played on (e.g. 'de_mirage')."""
    response = requests.get(f"{BASE_URL}/matches/{match_id}/stats", headers=headers)
    if response.status_code != 200:
        return 0
    return response.json().get("rounds", [])[0].get("round_stats", {}).get("Map", "0")


def get_num_rounds(match_id, headers):
    """Return the number of rounds played in the match."""
    response = requests.get(f"{BASE_URL}/matches/{match_id}/stats", headers=headers)
    if response.status_code != 200:
        return 0
    return response.json().get("rounds", [])[0].get("round_stats", {}).get("Rounds", "0")


def get_match_result(match_id, headers, player_name):
    """Return 1 if the player won the match, 0 otherwise."""
    response = requests.get(f"{BASE_URL}/matches/{match_id}", headers=headers)
    if response.status_code != 200:
        return 0

    match = response.json()
    player_faction = "faction2"
    for player in match.get("teams", {}).get("faction1", {}).get("roster", []):
        if player["nickname"] == player_name:
            player_faction = "faction1"
            break

    winner = match.get("detailed_results", [{}])[0].get("winner", None)
    return 1 if winner == player_faction else 0
