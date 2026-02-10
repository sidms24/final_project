#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 20:39:12 2026

@author: sofiahueffer
"""

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from probability_conversion import load_metrics

nba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/NBA_ODDS.csv"
wnba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/WNBA_ODDS.csv"

def test_market_efficiency(df, league_name="League"):
    
    df["score_diff"] = df.home_team_score - df.away_team_score
    df["std_diff"] = df["score_diff"].std()
    df["expected_margin"] = norm.ppf(df.PROF) * df["std_diff"]
    
    df = df.dropna(subset=["PROF","home_team_score","away_team_score"])
    
    X = sm.add_constant(df.expected_margin)
    y = df.score_diff 
    
    model = sm.OLS(y,X).fit()
    
    print(f"\n{league_name} Market Efficiency")
    print(model.summary())
    
    plt.figure(figsize=(8,5))
    plt.scatter(df.expected_margin, y, alpha=0.3)
    plt.plot(df.expected_margin, model.predict(X), color="red")
    plt.xlabel("Expected Point Differential (from PROB)")
    plt.ylabel("Actual Point Differential (Home - Away)")
    plt.title(f"{league_name} Market Efficiency")
    plt.show()
    
    return model


def efficiency_over_time(df, league_name="League", overall_slope=None):
    df = df.dropna(subset=["score_diff","expected_margin","year"])
    slopes = []
    years = sorted(df["year"].unique())
    
    for y in years:
        df_y = df[df["year"] == y]
        X = sm.add_constant(df_y["expected_margin"])
        y_val = df_y["score_diff"]
        model = sm.OLS(y_val, X).fit()
        slopes.append(model.params["expected_margin"])
    
    plt.figure(figsize=(8,5))
    plt.plot(years, slopes, marker='o')
    plt.axhline(1, color='red', linestyle='--', label='Efficient Market (slope=1)')
    plt.axhline(overall_slope, color='green', linestyle='-.', label=f'Overall Slope = {overall_slope:.2f}')
    plt.xlabel("Year")
    plt.ylabel("Regression Slope")
    plt.title(f"{league_name} Market Efficiency Over Time")
    plt.legend()
    plt.show()
    
    return years, slopes


if __name__ == "__main__":
    nba_df = load_metrics(nba_url)
    wnba_df = load_metrics(wnba_url, wnba=True)
    
    nba_model = test_market_efficiency(nba_df,"NBA")
    wnba_model = test_market_efficiency(wnba_df,"WNBA")
    
    years_nba, slopes_nba = efficiency_over_time(nba_df, "NBA", overall_slope=nba_model.params["expected_margin"])
    years_wnba, slopes_wnba = efficiency_over_time(wnba_df, "WNBA", overall_slope=wnba_model.params["expected_margin"])
