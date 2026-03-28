
import datetime as dt
import gc
import pandas as pd


def load_agency_metadata(nibrs_cutoff=None):
  """Load ORI-level agency type and population from FBI CDE API extract."""
  ag = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NIBRS_data/nibrs_pop_and_type.csv')
  ag['nibrs_start_date'] = pd.to_datetime(ag['nibrs_start_date'])
  # 1. "X City" → "X (City)"
  mask = ag['county'].str.endswith(' City')
  ag.loc[mask, 'county'] = ag.loc[mask, 'county'].str.replace(' City$', ' (City)', regex=True)
  ag['county'] = ag['county'].str.replace(r' County$', '', regex=True)
  ag['county'] = ag['county'].str.split(',')
  ag = ag.explode('county')
  ag['county'] = ag['county'].str.strip()
  ag['county'] = ag['county'].str.replace(r' \(City\)$', '', regex=True)
  # 2. Specific name variants
  manual_map = {
      'Lagrange': 'La Grange',
      'Lamoure': 'La Moure',
      'Leflore': 'Le Flore',
      'Ste Genevieve': 'St Genevieve',
      'Highlands': 'Highland',
      'Dekalb': 'De Kalb',
      'Desoto': 'De Soto',
      'Dewitt': 'De Witt',
      'Dupage': 'Du Page',
      'Carson (City)': 'Carson City (City)',
      'Charles (City)': 'Charles City',
      'James (City)': 'James City',
      'Prince Of Wales-Hyder': 'Prince Of Wales-Hyder Census Area'
  }
  ag['county'] = ag['county'].replace(manual_map)

  ag = ag[ag['is_nibrs'] == True]
  if nibrs_cutoff is not None:
      ag = ag[ag['nibrs_start_date'] <= nibrs_cutoff]
  ag = ag[ag['agency_type'].str.lower().str.contains('city|county', na=False)]

  ag = ag.drop_duplicates(subset=['ori', 'county', 'state', 'year'])
  ag.rename(columns={'population': 'ori_population'}, inplace=True)

  return ag


def load_ipv(hour_range=None, nibrs_cutoff=None):
  print('loading ipv')
  df = pd.read_parquet('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/cleaned_nibrs/dv_data_v2.parquet')
  df.drop(columns=['county_x', 'county_y'], inplace=True, errors='ignore')
  df = df[df['season'] >= 2012]

  df['county'] = (df['county']
    .str.replace(r' City And Borough$', '', regex=True)
    .str.replace(r' Municipality$', '', regex=True)
    .str.replace(r' County$', '', regex=True)
    .str.replace(r' \(City\)$', '', regex=True)
    .str.replace(r' City$', '', regex=True)
  )
  df['county'] = df['county'].replace({'James': 'James City'})

  df['incident_hour'] = (
    df['incident_date_hour']
    .str.extract(r'(\d{2}):\d{2} and')
    .astype(float)
  )
  df.loc[df['incident_date_hour'].str.contains('midnight', na=False), 'incident_hour'] = 24

  # ── Merge agency metadata ──
  print('loading metadata')
  ag = load_agency_metadata(nibrs_cutoff=nibrs_cutoff)
  print('merging ipv and metadata')
  df = df.merge(ag, on=['ori', 'county', 'state', 'year'], how='left')
  del ag
  gc.collect()

  df['used_alcohol'] = (df['offender_suspected_of_using_1'] == 'alcohol').astype(int)

  mask = (
        (df['vfemale'] == 1) &
        (df['ofemale'] == 0) &
        (df['location_type'] == 'residence/home') &
        (df['incident_hour'].notna())
    )
  df = df[mask].copy()
  del mask
  gc.collect()

  df['incident_hour'] = df['incident_hour'].astype(int)
  df['game_date'] = df['adjusted_date']

  # ── Hour-range filter and date shifting ──
  if hour_range is not None:
      start, end = hour_range
      if start < end:  # e.g. (12, 24) — same-day window
          df = df[df['incident_hour'].between(start, end - 1)]
      else:  # wraps midnight e.g. (18, 6) — overnight window
          df = df[(df['incident_hour'] >= start) | (df['incident_hour'] < end)]
          # Pre-dawn incidents belong to previous day's game
          df.loc[df['incident_hour'] < end, 'game_date'] -= pd.Timedelta(days=1)

  df['year'] = df['game_date'].dt.year.astype(str)
  df['is_bgfriend'] = (df['bgfriend'] == 1).astype(int)
  df['is_spouse']   = ((df['spouse'] == 1) | (df['commonspouse'] == 1)).astype(int)
  df['is_ipv']      = (df['intpartner'] == 1).astype(int)

  print('aggregating to daily counts')
  result = (
      df.groupby(['ori', 'county', 'state', 'game_date', 'year', 'ori_population',
                   'nibrs_start_date'], as_index=False)
      .agg(
          ipv_count      = ('is_ipv',      'sum'),
          spouse_count   = ('is_spouse',   'sum'),
          bgfriend_count = ('is_bgfriend', 'sum'),
          alcohol_count  = ('used_alcohol', 'sum'),
      )
  )
  del df
  gc.collect()

  return result
