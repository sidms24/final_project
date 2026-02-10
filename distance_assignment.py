#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 23:02:04 2026

@author: sofiahueffer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from probability_conversion import load_metrics, add_team_perspectives

state_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/state_locations.csv"
county_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/county_locations.csv"

nba_stadium_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/NBA_Stadiums.csv"
wnba_stadium_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/WNBA_Stadiums.csv"

nba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/NBA_ODDS.csv"
wnba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/WNBA_ODDS.csv"

def assign_state_favorite(states_df, teams_df, wnba=False):
    fav_teams = []
    for _, state in states_df.iterrows():
        lat_s, lon_s = state["latitude"], state["longitude"]
        distances = np.sqrt((teams_df["latitude"] - lat_s)**2 + (teams_df["longitude"] - lon_s)**2)
        closest_idx = distances.idxmin()
        team_name = teams_df.loc[closest_idx, "team"]
        fav_teams.append(team_name)
    states_df["favorite_team"] = fav_teams
    return states_df

def distance_states_to_loser(outcomes_df, state_favorites_df):
    rows = []
    for _, game in outcomes_df.iterrows():
        team = game.team
        for _, st in state_favorites_df[state_favorites_df["favorite_team"] == team].iterrows():
            rows.append({**game.to_dict(), "state": st["state"]})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    def run_dis_assign(url, type):
        states_df = pd.read_csv(url)
        nba_df    = pd.read_csv(nba_stadium_url)
        wnba_df   = pd.read_csv(wnba_stadium_url)
    
        states_nba  = assign_state_favorite(states_df.copy(), nba_df, wnba=False)
        states_wnba = assign_state_favorite(states_df.copy(), wnba_df, wnba=True)
    
        states_nba.to_csv(f"../datasets/{type}_NBA_distance_favorite.csv", index=False)
        states_wnba.to_csv(f"../datasets/{type}_WNBA_distance_favorite.csv", index=False)
        
        nba_outcomes = add_team_perspectives(load_metrics(nba_url))
        wnba_outcomes = add_team_perspectives(load_metrics(wnba_url, wnba=True))
        
        nba_states_assigned = distance_states_to_loser(nba_outcomes, states_nba)
        wnba_states_assigned = distance_states_to_loser(wnba_outcomes, states_wnba)
        
        nba_states_assigned.to_csv(f"../datasets/{type}_NBA_dis_game_assign.csv", index=False)
        wnba_states_assigned.to_csv(f"../datasets/{type}_WNBA_dis_game_assign.csv", index=False)
    
        print("NBA unexpected games:", len(nba_outcomes))
        print("NBA rows with states:", len(nba_states_assigned))
        print("NBA unique games after state assignment:", nba_states_assigned[['date','home_team','away_team']].drop_duplicates().shape[0])
    
        print("WNBA unexpected games:", len(wnba_outcomes))
        print("WNBA rows with states:", len(wnba_states_assigned))
        print("WNBA unique games after state assignment:", wnba_states_assigned[['date','home_team','away_team']].drop_duplicates().shape[0])

    run_dis_assign(state_url, "state")
    run_dis_assign(county_url, "county")
    
    