#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 22:15:54 2026

@author: sofiahueffer
"""

import pandas as pd
from probability_conversion import load_metrics, add_team_perspectives
import matplotlib.pyplot as plt

nba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/NBA_ODDS.csv"
wnba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/WNBA_ODDS.csv"

nba_state_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/State_NBA.csv"
wnba_state_1223_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/State_WNBA_1223.csv"
wnba_state_24_url   = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/State_WNBA_24.csv"

def assign_states(unexpected_df, state_df, league="nba", threshold=35):
    state_cols = [f"{league}_state", f"{league}_state_2", f"{league}_state_3"]
    perc_cols  = [f"{league}_perc", f"{league}_perc_2", f"{league}_perc_3"]
    
    rows = []
    for _, game in unexpected_df.iterrows():
        team = game.team
        for s_col, p_col in zip(state_cols, perc_cols):
            for _, st in state_df[state_df[s_col] == team].iterrows():
                perc = float(st[p_col].strip('%'))
                if perc >= threshold:
                    rows.append({**game.to_dict(),
                                 "state": st["state"],
                                 "perc": perc})
                    
    df = pd.DataFrame(rows)

    df["day"] = pd.to_datetime(
        df["date"].str.extract(r"(\d{1,2}\s\w+\s\d{4})")[0],
        format="%d %b %Y",
        errors="coerce").dt.date

    both_teams = (df.groupby(["day","state"]).filter(lambda g:
                  {g.iloc[0]["home_team"], g.iloc[0]["away_team"]}.issubset(set(g["team"])) ))
    
    multi_day = (df.groupby(["day","state"]).filter(lambda g: g["team"].nunique() > 1))
    
    both_teams = both_teams.assign(perc=both_teams["perc"])
    multi_day  = multi_day.assign(perc=multi_day["perc"])
    
    return df, both_teams, multi_day

def tolerance_games_plot():

    tolerances = range(1, 51, 1)
    
    nba_games  = add_team_perspectives(load_metrics(nba_url))
    wnba_games = add_team_perspectives(load_metrics(wnba_url, wnba=True))

    nba_main, wnba_main = [], []
    nba_all, nba_both, nba_multi = [], [], []
    wnba_all, wnba_both, wnba_multi = [], [], []

    w1223 = wnba_games[wnba_games.year <= 2023]
    w24   = wnba_games[wnba_games.year == 2024]

    for t in tolerances:
        df, b, m = assign_states(nba_games, nba_state, "nba", t)
        nba_all.append(len(df))
        nba_both.append(len(b))
        nba_multi.append(len(m))
        nba_main.append(len(df) - len(b) - len(m))

        df1, b1, m1 = assign_states(w1223, wnba_state_1223, "wnba", t)
        df2, b2, m2 = assign_states(w24, wnba_state_24, "wnba", t)
        df_total = pd.concat([df1, df2])
        b_total = pd.concat([b1, b2])
        m_total = pd.concat([m1, m2])
        wnba_all.append(len(df_total))
        wnba_both.append(len(b_total))
        wnba_multi.append(len(m_total))
        wnba_main.append(len(df_total) - len(b_total) - len(m_total))

    plt.figure(figsize=(8,5))
    plt.plot(tolerances, nba_main, color="navy", linewidth=2.5, label="NBA usable games")
    plt.plot(tolerances, wnba_main, color="darkorange", linewidth=2.5, label="WNBA usable games")
    plt.xlabel("Tolerance (%)")
    plt.ylabel("Number of Games")
    plt.title("NBA vs WNBA Usable Games")
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    axes[0].plot(tolerances, nba_all,   color="tab:blue",  alpha=0.7, label="All rows")
    axes[0].plot(tolerances, nba_both,  color="tab:blue",  alpha=0.7, linestyle=":", label="Both teams same game")
    axes[0].plot(tolerances, nba_multi, color="tab:blue",  alpha=0.7, linestyle="--", label="Multi-team same day")
    axes[0].plot(tolerances, nba_main,  color="navy",      alpha=1.0, linewidth=2.5, label="Usable games")
    axes[0].set_title("NBA")
    axes[0].set_xlabel("Tolerance (%)")
    axes[0].set_ylabel("Number of Games")
    axes[0].legend(frameon=False, loc='upper right')

    axes[1].plot(tolerances, wnba_all,   color="tab:orange", alpha=0.7, label="All rows")
    axes[1].plot(tolerances, wnba_both,  color="tab:orange", alpha=0.7, linestyle=":", label="Both teams same game")
    axes[1].plot(tolerances, wnba_multi, color="tab:orange", alpha=0.7, linestyle="--", label="Multi-team same day")
    axes[1].plot(tolerances, wnba_main,  color="darkorange", alpha=1.0, linewidth=2.5, label="Usable games")
    axes[1].set_title("WNBA")
    axes[1].set_xlabel("Tolerance (%)")
    axes[1].set_ylabel("Number of Games")
    axes[1].legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nba_state       = pd.read_csv(nba_state_url)
    wnba_state_1223 = pd.read_csv(wnba_state_1223_url)
    wnba_state_24   = pd.read_csv(wnba_state_24_url)
    
    tolerance_games_plot()
    
    """
    nba_team_perspectives  = add_team_perspectives(load_metrics(nba_url))
    wnba_team_perspectives = add_team_perspectives(load_metrics(wnba_url, wnba=True))
    
    nba_states_assigned, nba_both, nba_multi = assign_states(nba_team_perspectives, nba_state, league="nba")
    
    wnba_1223 = wnba_team_perspectives[wnba_team_perspectives["year"] <= 2023]
    wnba_24   = wnba_team_perspectives[wnba_team_perspectives["year"] == 2024]
    
    wnba_states_1223, wnba_both_1223, wnba_multi_1223 = assign_states(wnba_1223, wnba_state_1223, league="wnba")
    wnba_states_24, wnba_both_24, wnba_multi_24 = assign_states(wnba_24, wnba_state_24, league="wnba")
    
    wnba_states_assigned = pd.concat([wnba_states_1223, wnba_states_24], ignore_index=True)
    wnba_both  = pd.concat([wnba_both_1223,  wnba_both_24],  ignore_index=True)
    wnba_multi = pd.concat([wnba_multi_1223, wnba_multi_24], ignore_index=True)
    
    nba_states_assigned.to_csv("../datasets/NBA_outcomes_with_states.csv", index=False)
    wnba_states_assigned.to_csv("../datasets/WNBA_outcomes_with_states.csv", index=False)
    
    nba_both.to_csv("../datasets/NBA_both_teams.csv", index=False)
    nba_multi.to_csv("../datasets/NBA_multi_day.csv", index=False)
    wnba_both.to_csv("../datasets/WNBA_both_teams.csv", index=False)
    wnba_multi.to_csv("../datasets/WNBA_multi_day.csv", index=False)
    
    print("NBA games:", int(len(nba_team_perspectives) / 2))
    print("NBA rows with states:", len(nba_states_assigned))
    print("NBA % both teams:", round(100 * len(nba_both) / len(nba_states_assigned), 2))
    print("NBA % multi day:",  round(100 * len(nba_multi) / len(nba_states_assigned), 2))
    
    print("WNBA games:", int(len(wnba_team_perspectives) / 2))
    print("WNBA rows with states:", len(wnba_states_assigned))
    print("WNBA % both teams:", round(100 * len(wnba_both) / len(wnba_states_assigned), 2))
    print("WNBA % multi day:",  round(100 * len(wnba_multi) / len(wnba_states_assigned), 2))
    """
    