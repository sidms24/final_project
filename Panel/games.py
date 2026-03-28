import pandas as pd
import string
import datetime as dt


def load_game_outcomes(league='NBA', trends=False):
  if league.lower() == 'nba' and not trends:
    game_outcomes = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NBA/NBA_game_outcomes.csv')
  elif league.lower() == 'wnba' and not trends:
    game_outcomes = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/WNBA/WNBA_game_outcomes.csv')
    for col in ['home_team', 'away_team', 'team', 'winner']:
      game_outcomes[col] = game_outcomes[col].str[:-2]
  elif league.lower() == 'nba' and trends:
    game_outcomes = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NBA/NBA_outcomes_with_states.csv')
    game_outcomes['state'] = game_outcomes['state'].apply(lambda x: string.capwords(x))
  elif league.lower() == 'wnba' and trends:
    game_outcomes = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/WNBA/WNBA_outcomes_with_states.csv')
    game_outcomes['state'] = game_outcomes['state'].apply(lambda x: string.capwords(x))
    for col in ['home_team', 'away_team', 'winner']:
      game_outcomes[col] = game_outcomes[col].str[:-2]
  game_outcomes['dt_obj'] = pd.to_datetime(game_outcomes['date'].apply(lambda x: x.split(' -')[0]), format='mixed', errors='coerce')
  game_outcomes['game_date'] = (game_outcomes['dt_obj'] - dt.timedelta(hours=5)).dt.normalize()
  game_outcomes.drop(columns=['date', 'dt_obj'], inplace=True)
  base_cols = ['game_date','home_team', 'away_team','team' ,'game_outcome', 'winner', 'bookmakers', 'time']
  # Include team_prob for implied-probability model (Card & Dahl specification)
  optional = ['team_prob', 'predwin', 'predclose', 'predloss', 'win', 'loss']
  cols = base_cols + [c for c in optional if c in game_outcomes.columns]
  if trends:
    cols += ['state']
  return game_outcomes[cols]