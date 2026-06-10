"""Fetch a player's match history from the FACEIT API into an Excel file.

For every recent match of the given player, the script collects the raw
inputs the model is built from (kills, enemy team elo, time of day, rounds,
deaths, ADR, map, result, player elo) and saves them to
data/LARGE_DATA_<player>.xlsx.

Requires the FACEIT_API_KEY environment variable to be set.
"""

import os

import pandas as pd

import getmatches

API_KEY = os.environ["FACEIT_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}


def main():
    player_name = input("Enter player name: ")
    num_matches = int(input("Enter number of matches: "))

    matches_id = getmatches.get_player_matches(player_name, headers, num_matches)

    # The player's own elo is fetched once since it barely changes
    # over the collection window.
    player_elo = getmatches.get_elo(player_name, headers)

    output_path = f"data/LARGE_DATA_{player_name}.xlsx"
    df = pd.DataFrame(
        columns=[
            "kills", "enemy_team_elo", "time", "rounds",
            "deaths", "adr", "map", "won", "player_elo",
        ]
    )

    for i, match_id in enumerate(matches_id, start=1):
        print(f"Match {i}/{len(matches_id)}")

        row = {
            "kills": getmatches.get_player_kills(player_name, match_id, headers),
            "enemy_team_elo": getmatches.get_enemy_team_elos(player_name, match_id, headers),
            "time": getmatches.get_match_time(match_id, headers),
            "rounds": getmatches.get_num_rounds(match_id, headers),
            "deaths": getmatches.get_player_deaths(player_name, match_id, headers),
            "adr": getmatches.get_player_damage(player_name, match_id, headers),
            "map": getmatches.get_match_map(match_id, headers),
            "won": getmatches.get_match_result(match_id, headers, player_name),
            "player_elo": player_elo,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Checkpoint every 50 matches so a failed request doesn't lose everything.
        if i % 50 == 0:
            df.to_excel(output_path, index=False)

    df.to_excel(output_path, index=False)
    print(f"Saved {len(df)} matches to {output_path}")


if __name__ == "__main__":
    main()
