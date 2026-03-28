#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:32:07 2026

@author: sofiahueffer
"""

import pandas as pd
import numpy as np

nba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/NBA_ODDS.csv"
wnba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/WNBA_ODDS.csv"

def convert_odds(odds, wnba=False):
    odds = np.array(odds, float)
    if wnba:
        odds = np.where(odds==-100,-99.999,odds)
        odds = np.where(odds==100,99.999,odds)
        prob = np.where(odds>0,100/(odds+100),-odds/(-odds+100))
    else:
        prob = 1 / odds
    return prob

def load_metrics(url, wnba=False):
    df = pd.read_csv(url)
    
    df["year"] = df["date"].str.extract(r"(\d{4})").astype(int)
    df = df[(df.year>=2012)&(df.year<=2024)]
    
    df = df.dropna(subset=["odds1","odds2","home_team_score","away_team_score"])
    
    for col in ["odds1","odds2"]:
        df[col] = convert_odds(df[col], wnba=wnba)

    df["PROF"]  = df["odds1"]/(df["odds1"]+df["odds2"])
    df["SPR"] = (1/df["odds1"])+(1/df["odds2"])-1
    
    return df

def add_team_perspectives(df):
    rows = []
    for _, g in df.iterrows():
        home_prob = g.PROF
        away_prob = 1 - g.PROF
        home_win  = g.home_team_score > g.away_team_score
        away_win  = not home_win

        def outcome(p,w):
            return ("unexpected_loss" if not w and p>=0.67 else
                    "close_loss"      if not w and p>=0.5 else
                    "expected_loss"   if not w else
                    "unexpected_win"  if w and p<=0.33 else
                    "close_win"       if w and p<0.5 else
                    "expected_win")

        def classify(p, w):
            """Three-category implied-probability classification (Card & Dahl)."""
            return {
                'predwin':   int(p >= 0.67),
                'predclose': int(0.33 < p < 0.67),
                'predloss':  int(p <= 0.33),
                'win':       int(w),
                'loss':      int(not w),
            }

        rows.append({**g, "team": g.home_team, "team_prob": home_prob,
                     "game_outcome": outcome(home_prob, home_win),
                     **classify(home_prob, home_win)})
        rows.append({**g, "team": g.away_team, "team_prob": away_prob,
                     "game_outcome": outcome(away_prob, away_win),
                     **classify(away_prob, away_win)})

    return pd.DataFrame(rows)

if __name__=="__main__":
    nba  = add_team_perspectives(load_metrics(nba_url))
    wnba = add_team_perspectives(load_metrics(wnba_url, wnba=True))

    nba.to_csv("../datasets/NBA_game_outcomes.csv", index=False)
    wnba.to_csv("../datasets/WNBA_game_outcomes.csv", index=False)
