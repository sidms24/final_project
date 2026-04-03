import pandas as pd
import string
import datetime as dt

def _option_b_outcome(p, won, sd_band):
  midpoint = 0.50
  lower = midpoint - sd_band
  upper = midpoint + sd_band

  if won:
    if p <= lower:
      return 'unexpected_win'
    elif p <= midpoint:
      return 'close_win'
    else:
      return 'expected_win'
  else:
    if p >= upper:
      return 'unexpected_loss'
    elif p >= midpoint:
      return 'close_loss'
    else:
      return 'expected_loss'


def load_game_outcomes(league='NBA', trends=False, recompute_option_b=True):
  if league.lower() == 'nba' and not trends:
    game_outcomes = pd.read_csv(
      'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NBA/NBA_game_outcomes.csv'
    )

  elif league.lower() == 'wnba' and not trends:
    game_outcomes = pd.read_csv(
      'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/WNBA/WNBA_game_outcomes.csv'
    )
    for col in ['home_team', 'away_team', 'team', 'winner']:
      if col in game_outcomes.columns:
        game_outcomes[col] = game_outcomes[col].str[:-2]

  elif league.lower() == 'nba' and trends:
    game_outcomes = pd.read_csv(
      'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NBA/NBA_outcomes_with_states.csv'
    )
    game_outcomes['state'] = game_outcomes['state'].apply(lambda x: string.capwords(x))

  elif league.lower() == 'wnba' and trends:
    game_outcomes = pd.read_csv(
      'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/WNBA/WNBA_outcomes_with_states.csv'
    )
    game_outcomes['state'] = game_outcomes['state'].apply(lambda x: string.capwords(x))
    for col in ['home_team', 'away_team', 'team', 'winner']:
      if col in game_outcomes.columns:
        game_outcomes[col] = game_outcomes[col].str[:-2]

  else:
    raise ValueError("league must be 'NBA' or 'WNBA'")

  # Recompute game_outcome using Option B
  if recompute_option_b:
    if 'team_prob' not in game_outcomes.columns:
      raise ValueError("Option B requires a 'team_prob' column, but it is missing.")

    # Estimate SD from the actual sample of team probabilities
    sd_band = game_outcomes['team_prob'].dropna().std()

    # Determine whether this row's team won
    if 'win' in game_outcomes.columns:
      won = game_outcomes['win'].astype(bool)
    else:
      won = game_outcomes['team'] == game_outcomes['winner']

    game_outcomes['game_outcome'] = [
      _option_b_outcome(p, w, sd_band)
      for p, w in zip(game_outcomes['team_prob'], won)
    ]

  game_outcomes['dt_obj'] = pd.to_datetime(
    game_outcomes['date'].apply(lambda x: x.split(' -')[0]),
    format='mixed',
    errors='coerce'
  )
  game_outcomes['game_date'] = (game_outcomes['dt_obj'] - dt.timedelta(hours=5)).dt.normalize()
  game_outcomes.drop(columns=['date', 'dt_obj'], inplace=True)

  base_cols = ['game_date', 'home_team', 'away_team', 'team', 'game_outcome',
               'winner', 'bookmakers', 'time']
  optional = ['team_prob', 'predwin', 'predclose', 'predloss', 'win', 'loss']
  cols = base_cols + [c for c in optional if c in game_outcomes.columns]

  if trends:
    cols += ['state']

  return game_outcomes[cols]
