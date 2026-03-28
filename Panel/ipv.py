
import pandas as pd


def load_agency_metadata():
  """Load ORI-level agency type and population from FBI CDE API extract."""
  ag = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/NIBRS_data/nibrs_pop_and_type.csv')
  # One row per ORI: take most recent year's data
  ag = ag.sort_values('year').drop_duplicates(subset='ori', keep='last')
  # Normalise county name to title case (source is ALL CAPS)
  ag['county'] = ag['county'].str.title()
  return ag

def load_ipv():
  df = pd.read_parquet('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/cleaned_nibrs/dv_data_v2.parquet')

  # ── Merge agency metadata for type filter ──
  ag = load_agency_metadata()
  n_before = df['ori'].nunique()
  df = df.merge(ag[['ori', 'agency_type', 'population', 'nibrs_start_date']], on='ori', how='left')
  n_matched = df['agency_type'].notna().sum()
  print(f"Agency metadata merge: {n_matched:,}/{len(df):,} incidents matched "
        f"({df['ori'].nunique()}/{n_before} ORIs)")

  # Exclude Shelby County, TN — systematic miscoding of relationship categories
  # (Cardazzi et al. 2024: 61.5% bf/gf vs 45% avg, underuses "relationship unknown")
  shelby_mask = (df['county'].str.lower() == 'shelby') & (df['state'].str.lower() == 'tennessee')
  n_shelby = shelby_mask.sum()
  df = df[~shelby_mask].copy()
  if n_shelby > 0:
      print(f"Shelby County exclusion: dropped {n_shelby:,} incidents (Cardazzi et al. 2024)")

  # Exclude state police, college police, and special agencies (Card & Dahl 2011)
  excluded_types = ['state police', 'college', 'university', 'special']
  type_mask = df['agency_type'].str.lower().str.contains(
      '|'.join(excluded_types), na=False)
  n_excluded = type_mask.sum()
  df = df[~type_mask].copy()
  print(f"Agency type exclusion: dropped {n_excluded:,} non-city/county incidents (Card & Dahl 2011)")
  df.drop(columns=['agency_type'], inplace=True)
  df.rename(columns={'population': 'ori_population'}, inplace=True)
  df['nibrs_start_date'] = pd.to_datetime(df['nibrs_start_date'], errors='coerce')

  mask = (
        (df['vfemale'] == 1) &
        (df['ofemale'] == 0) &
        (df['location_type'] == 'residence/home') &
        (df['incident_hour'].notna())
    )
  df = df[mask].copy()
  df['incident_hour'] = df['incident_hour'].astype(int)
  df['used_alcohol'] = (df['offender_suspected_of_using_1'] == 'alcohol').astype(int)
  df['game_date'] = df['adjusted_date'].where(
      df['incident_hour'] >= 6,
      df['adjusted_date'] - pd.Timedelta(days=1)
  )
  df['year'] = df['game_date'].dt.year.astype(str)
  df['is_bgfriend'] = (df['bgfriend'] == 1).astype(int)
  df['is_spouse']   = ((df['spouse'] == 1) | (df['commonspouse'] == 1)).astype(int)
  df['is_ipv']      = (df['intpartner'] == 1).astype(int)
  result = (
      df.groupby(['ori', 'county', 'state', 'game_date', 'year', 'ori_population',
                   'nibrs_start_date', 'incident_hour'], as_index=False)
      .agg(
          ipv_count      = ('is_ipv',      'sum'),
          spouse_count   = ('is_spouse',   'sum'),
          bgfriend_count = ('is_bgfriend', 'sum'),
          alcohol_count  = ('used_alcohol', 'sum'),
      )
  )
  return result

