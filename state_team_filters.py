#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:22:04 2026

@author: sofiahueffer
"""

from huggingface_hub import list_repo_files
import pandas as pd

## ----- Unique Teams ----- ##

def get_team_stats(url, min_games=50):
    df = pd.read_csv(url)
    df['clean_date'] = pd.to_datetime(df['date'].str.extract(r'(\d{1,2}\s\w+\s\d{4})')[0], format='%d %b %Y', errors='coerce')
    all_teams = pd.concat([
        df[['home_team','clean_date']].rename(columns={'home_team':'team','clean_date':'date'}),
        df[['away_team','clean_date']].rename(columns={'away_team':'team','clean_date':'date'})
    ]).dropna()
    stats = all_teams.groupby('team')['date'].agg(['min','max','count']).reset_index()
    stats = stats[stats['count'] >= min_games]
    stats['period'] = stats['min'].dt.strftime('%d %b %Y') + ' - ' + stats['max'].dt.strftime('%d %b %Y')
    print(len(stats['team']))
    return stats[['team','period','count']]

files = list_repo_files("sidms/Final_project", repo_type="dataset")
#print(files)

nba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/NBA/NBA_ODDS.csv"
wnba_url = "https://huggingface.co/datasets/group-a/Final_project/resolve/main/WNBA/WNBA_ODDS.csv"

print(get_team_stats(nba_url))
print(get_team_stats(wnba_url))


## ----- Unique States ----- ##

start_year, end_year, min_freq = 2012, 2024, 100
url = lambda y: f"https://huggingface.co/datasets/group-a/Final_project/resolve/main/NIBRS_data/dv_data_{y}.csv"

df_start = pd.read_csv(url(start_year), usecols=['state'])
df_end = pd.read_csv(url(end_year), usecols=['state'])

common = set(df_start['state'].dropna()) & set(df_end['state'].dropna())

freq_start = df_start[df_start['state'].isin(common)].value_counts('state')
freq_end = df_end[df_end['state'].isin(common)].value_counts('state')

result = pd.DataFrame({
    'state': sorted(common),
    f'freq_{start_year}': freq_start,
    f'freq_{end_year}': freq_end
}).fillna(0)

result[[f'freq_{start_year}', f'freq_{end_year}']] = result[[f'freq_{start_year}', f'freq_{end_year}']].astype(int)

result = result[result[f'freq_{start_year}'] >= min_freq]

print(result, len(result))
