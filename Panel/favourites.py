import pandas as pd

def load_favourites(league='nba'):
  if league.lower() == 'nba':
    df = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NBA/county_NBA_distance_favorite.csv')
  elif league.lower() == 'wnba':
    df = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/WNBA/county_WNBA_distance_favorite.csv')
  df = df.drop(columns=['Unnamed: 0', 'longitude', 'latitude'])
  df = df.rename(columns={'state': 'county', 'favorite_team':'team'})
  if league.lower() == 'wnba':
    df['team'] = df['team'].str.replace(r'\s*W$', '', regex=True)
  mapping = df.set_index('county')['team'].to_dict()
  return mapping