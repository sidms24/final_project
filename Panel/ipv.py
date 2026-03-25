
import pandas as pd
def load_ipv():
  df = pd.read_parquet('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/cleaned_nibrs/dv_data_v2.parquet')
  mask = (
        (df['vfemale'] == 1) &
        (df['ofemale'] == 0) &
        (df['location_type'] == 'residence/home') &
        (df['incident_hour'].notna())
    )
  df = df[mask].copy()
  df['incident_hour'] = df['incident_hour'].astype(int)
  df = df[(df['incident_hour'] >= 18) | (df['incident_hour'] < 6)]
  df['game_date'] = df['adjusted_date'].where(
      df['incident_hour'] >= 6,
      df['adjusted_date'] - pd.Timedelta(days=1)
  )
  df['year'] = df['game_date'].dt.year.astype(str)
  df['is_bgfriend'] = (df['bgfriend'] == 1).astype(int)
  df['is_spouse']   = ((df['spouse'] == 1) | (df['commonspouse'] == 1)).astype(int)
  df['is_ipv']      = (df['intpartner'] == 1).astype(int)
  result = (
      df.groupby(['ori', 'county', 'state', 'game_date', 'year'], as_index=False)
      .agg(
          ipv_count      = ('is_ipv',      'sum'),
          spouse_count   = ('is_spouse',   'sum'),
          bgfriend_count = ('is_bgfriend', 'sum'),
      )
  )
  return result

