
import pandas as pd

def load_legalisation(league, trends=False):
  if trends:
    HF_BASE = 'https://huggingface.co/datasets/group-a/Final_Project/resolve/main'
    legal = pd.read_csv(f'{HF_BASE}/US_data/legalisation_all_states.csv')
  elif league.lower() == 'nba':
    legal = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NBA/legalisation.csv')
  elif league.lower() == 'wnba':
    legal = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/WNBA/wnba_legalisation.csv')
  for col in ['date', 'OSB_date', 'announced']:
      if col in legal.columns:
          legal[col] = pd.to_datetime(legal[col], errors='coerce')
  legal.rename(columns={'date': 'legal_date'}, inplace=True)
  keep = [c for c in ['team', 'legal_date', 'OSB_date', 'announced',
                        'sports_betting_legal', 'betting_type', 'state'] if c in legal.columns]
  legal['legal_date'] = pd.to_datetime(legal['legal_date'])
  return legal[keep]