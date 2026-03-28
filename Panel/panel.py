from .confounders import load_confounders, load_game_controls, load_handle
from .favourites import load_favourites
from .games import load_game_outcomes
from .ipv import load_ipv
from .policy import load_legalisation
from .CONSTANTS import state_map_nba, state_map_wnba, county_map_nba, county_map_wnba
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

class GamePanel:
    def __init__(self, league, county_team_mapping=None, grey_zone=False,
                 trends=False, county_only=False, hour_range=(18, 6),
                 min_coverage=0.0, min_seasons=0, pop_cap=None,
                 nibrs_cutoff=None):
      self.league = league
      self.hour_range = hour_range
      self.trends = trends
      self.grey_zone = grey_zone
      self.county_only = county_only
      self.min_coverage = min_coverage
      self.min_seasons = min_seasons
      self.pop_cap = pop_cap
      self.nibrs_cutoff = nibrs_cutoff
      if league.lower() == 'nba':
        self.state_map = state_map_nba
        self.county_map = county_map_nba
      else: 
         self.county_map = county_map_wnba
         self.state_map = state_map_wnba
       
      if county_team_mapping is None:
        self.mapping = load_favourites(league)
        self.counties = list(self.mapping.keys())
      else:
        self.mapping = county_team_mapping
        self.counties = list(self.mapping.keys())
      self._active_days = None
      self._controls = None
      self._confounders = None
      self._outcomes = None
      self._games = None
      self._legal = None
      self._ipv = None
      self._panel = None
      self._handle = None

    def load_game_controls(self):
      if self._controls is None: self._controls = load_game_controls(self.league)
      return self._controls
    def load_confounders(self):
      if self._confounders is None: self._confounders = load_confounders(self.league)
      return self._confounders
    def load_game_outcomes(self):
      if self._outcomes is None: self._outcomes = load_game_outcomes(self.league, self.trends)
      return self._outcomes
    def load_games(self):
      if self._games is None:
        self._games = self.load_game_outcomes().merge(
            self.load_game_controls(), on=['game_date', 'home_team', 'away_team'], how='inner')
        seasons = self._games.groupby('season')['game_date'].agg(['min', 'max']).reset_index()
        seasons['date_range'] = seasons.apply(
            lambda x: pd.date_range(start=x['min'], end=x['max']).tolist(), axis=1)
        self._active_days = set(day for sublist in seasons['date_range'] for day in sublist)
      return self._games
    def load_legalisation(self):
      if self._legal is None: self._legal = load_legalisation(self.league, self.trends)
      return self._legal
    def load_ipv(self):
        if self._ipv is None:
            self._ipv = load_ipv(hour_range=self.hour_range,
                                 nibrs_cutoff=self.nibrs_cutoff)
        return self._ipv
    def load_handle(self):
      if self._handle is None: self._handle = load_handle()
      return self._handle

    def _zero_fill(self, ipv):
      ipv = ipv.copy()
      ipv['game_date'] = pd.to_datetime(ipv['game_date'])
      ori_ref = (ipv.groupby(['ori', 'county', 'state'])
          .agg(min_date=('game_date', 'min'), max_date=('game_date', 'max'),
               nibrs_start_date=('nibrs_start_date', 'first'))
          .reset_index())
      # Use nibrs_start_date as floor so we don't zero-fill before the agency reported
      ori_ref['nibrs_start_date'] = pd.to_datetime(ori_ref['nibrs_start_date'])
      has_nibrs = ori_ref['nibrs_start_date'].notna()
      n_would_truncate = (has_nibrs & (ori_ref['nibrs_start_date'] > ori_ref['min_date'])).sum()
      ori_ref.loc[has_nibrs, 'min_date'] = ori_ref.loc[has_nibrs, [
          'nibrs_start_date', 'min_date']].max(axis=1)
      # Drop ORIs whose NIBRS start date is after their last observed incident
      empty = ori_ref['min_date'] > ori_ref['max_date']
      if empty.any():
          print(f"NIBRS left-truncation: dropped {empty.sum():,} ORIs with no valid date range "
                f"(nibrs_start_date > last incident)")
          ori_ref = ori_ref[~empty].reset_index(drop=True)
      print(f"NIBRS left-truncation: {has_nibrs.sum():,} ORIs with start date, "
            f"{n_would_truncate:,} left-truncated, {len(ori_ref):,} ORIs remaining")
      all_dates = np.sort(np.array(list(self._active_days), dtype='datetime64[ns]'))
      all_dates = all_dates[all_dates >= np.datetime64('2012-01-01')]
      start_idx = np.searchsorted(all_dates, ori_ref['min_date'].values.astype('datetime64[ns]'), side='left')
      end_idx   = np.searchsorted(all_dates, ori_ref['max_date'].values.astype('datetime64[ns]'), side='right')
      counts    = end_idx - start_idx
      total     = counts.sum()
      rep_start   = np.repeat(start_idx, counts)
      cum_counts  = np.cumsum(counts)
      group_base  = np.repeat(cum_counts - counts, counts)
      offsets     = np.arange(total) - group_base
      date_idx    = rep_start + offsets
      ori_row_idx = np.repeat(np.arange(len(ori_ref)), counts)
      grid = pd.DataFrame({
          'game_date': all_dates[date_idx], 'ori': ori_ref['ori'].values[ori_row_idx],
          'county': ori_ref['county'].values[ori_row_idx], 'state': ori_ref['state'].values[ori_row_idx],
          'nibrs_start_date': ori_ref['nibrs_start_date'].values[ori_row_idx],
      })
      # ori_population can vary by year — use median per ORI-county as stable estimate
      ori_pop = ipv.groupby(['ori', 'county'])['ori_population'].median()
      grid = grid.merge(ipv[['ori', 'county', 'game_date', 'ipv_count', 'spouse_count', 'bgfriend_count', 'alcohol_count']],
          on=['ori', 'county', 'game_date'], how='left')
      grid['ori_population'] = grid.set_index(['ori', 'county']).index.map(ori_pop)
      grid.reset_index(drop=True, inplace=True)
      for col in ['ipv_count', 'spouse_count', 'bgfriend_count', 'alcohol_count']:
          grid[col] = grid[col].fillna(0).astype(np.int32)
      grid['year'] = grid['game_date'].dt.year.astype(str)
      print(f"Zero-fill: {len(ori_ref):,} ORIs, {len(ipv):,} incident rows -> {total:,} panel rows")
      return grid
    
    def panel(self):
      games = self.load_games()
      confounders = self.load_confounders()
      legal = self.load_legalisation()
      ipv   = self.load_ipv()
      handle = self.load_handle()

      # ── Pre-zero-fill filters (reduce ipv before expensive grid expansion) ──
      n_ori_raw = ipv['ori'].nunique()

      if not self.trends:
          ipv = ipv[ipv['county'].isin(self.counties)]

      if self.grey_zone:
          _team = ipv['county'].map(self.mapping)
          ipv = ipv[_team.map(self.state_map) == ipv['state']]

      if self.county_only:
          _team = ipv['county'].map(self.mapping)
          ipv = ipv[_team.map(self.county_map) == ipv['county']]

      if self.pop_cap is not None:
          ipv = ipv[ipv['ori_population'] <= self.pop_cap]

      _cov_data = None
      if self.min_coverage > 0:
          if self.league.lower() == 'nba':
              cov_url = 'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/cleaned_nibrs/ori_game_day_reporting_nba.parquet'
          else:
              cov_url = 'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/cleaned_nibrs/ori_game_day_reporting_wnba.parquet'
          _cov_data = pd.read_parquet(cov_url)[['ori', 'season', 'reporting_frac']]
          _cov_data.rename(columns={'reporting_frac': 'coverage'}, inplace=True)
          _good_oris = _cov_data[_cov_data['coverage'] >= self.min_coverage]['ori'].unique()
          ipv = ipv[ipv['ori'].isin(_good_oris)]

      print(f"Pre-filters: {n_ori_raw:,} -> {ipv['ori'].nunique():,} ORIs before zero-fill")

      grid = self._zero_fill(ipv)

      if not self.trends and 'team' in games.columns:
          dupes = games.duplicated(subset=['team', 'game_date'], keep=False)
          if dupes.any():
              print(f"WARNING: {dupes.sum()} duplicate team-date rows — deduplicating")
              games = games.drop_duplicates(subset=['team', 'game_date'], keep='first')

      grid = grid.rename(columns={'ori_population': 'population_estimate'})
      # Drop rows before ORI started NIBRS reporting (structural missing, not true zeros)
      pre_nibrs = grid['nibrs_start_date'].notna() & (grid['game_date'] < grid['nibrs_start_date'])
      if pre_nibrs.any():
          print(f"Dropping {pre_nibrs.sum():,} pre-NIBRS-reporting rows ({pre_nibrs.sum()/len(grid)*100:.1f}%)")
          grid = grid[~pre_nibrs].copy()
      grid.drop(columns=['nibrs_start_date'], inplace=True)
      print(f"ORI-day panel: {grid['ori'].nunique():,} ORIs, {grid['county'].nunique():,} counties, {len(grid):,} rows")

      legal_state = (legal.drop(columns=['team', 'betting_type'], errors='ignore')
                          .drop_duplicates(subset=['state']))
      panel = grid.merge(legal_state, on='state', how='left')
      panel['policy'] = (panel['legal_date'].notna() & (panel['game_date'] >= panel['legal_date'])).astype(int)
      if 'OSB_date' in panel.columns:
          panel['policy_online'] = (panel['OSB_date'].notna() & (panel['game_date'] >= panel['OSB_date'])).astype(int)
      if {'announced', 'legal_date'}.issubset(panel.columns):
          m = (panel['announced'].notna() & panel['legal_date'].notna() &
               (panel['game_date'] >= panel['announced']) & (panel['game_date'] < panel['legal_date']))
          panel['announced_pending'] = m.astype(int)
      panel['cohort_year'] = panel['legal_date'].dt.year.fillna(10000).astype(int)

      # Game merge
      panel['team'] = panel['county'].map(self.mapping)
      if self.trends:
        outcome_rank = {'unexpected_loss': 0, 'close_loss': 1, 'expected_loss': 2,
                        'close_win': 3, 'expected_win': 4, 'unexpected_win': 5}
        games_st = games.copy()
        games_st['_rank'] = games_st['game_outcome'].map(outcome_rank)
        # Sort so worst outcome comes first — 'first' in agg picks that game's teams
        games_st = games_st.sort_values('_rank')
        agg_dict = {'game_outcome': 'first', '_rank': 'min', 'national_tv': 'max',
                    'is_playoff': 'max', 'PACE': 'mean', 'tipoff_hour': 'mean'}
        for col in ['home_team', 'away_team']:
            if col in games_st.columns:
                agg_dict[col] = 'first'
        agg_dict = {k: v for k, v in agg_dict.items() if k in games_st.columns}
        state_games = games_st.groupby(['state', 'game_date']).agg(agg_dict).reset_index()
        rank_to_outcome = {v: k for k, v in outcome_rank.items()}
        state_games['game_outcome'] = state_games['_rank'].map(rank_to_outcome)
        state_games.drop(columns='_rank', inplace=True)
        season_map = games_st.groupby(['state', 'game_date'])['season'].first().reset_index()
        state_games = state_games.merge(season_map, on=['state', 'game_date'], how='left')
        game_counts = games_st.groupby(['state', 'game_date']).size().rename('n_games_in_state')
        state_games = state_games.merge(game_counts, on=['state', 'game_date'], how='left')
        state_games['multiple_games'] = (state_games['n_games_in_state'] > 1).astype(int)
        panel = panel.merge(state_games, on=['state', 'game_date'], how='left')
      else:
        panel = panel.merge(games, on=['team', 'game_date'], how='left')

      panel['game_day'] = panel['game_outcome'].notna().astype(int)
      panel['game_outcome'] = panel['game_outcome'].fillna('no_game')

      # Confounder merge
      if 'team' in panel.columns:
          conf_home = confounders.rename(columns={'conf_attendance': 'conf_attendance_h',
              'avg_ref_rest_days': 'avg_ref_rest_days_h', 'rest_category': 'rest_category_h'})
          panel = panel.merge(conf_home[['game_date', 'home_team', 'conf_attendance_h',
              'avg_ref_rest_days_h', 'rest_category_h']],
              left_on=['game_date', 'team'], right_on=['game_date', 'home_team'], how='left')
          panel.drop(columns=['home_team_y'], errors='ignore', inplace=True)
          if 'home_team_x' in panel.columns: panel.rename(columns={'home_team_x': 'home_team'}, inplace=True)
          conf_away = confounders.rename(columns={'conf_attendance': 'conf_attendance_a',
              'avg_ref_rest_days': 'avg_ref_rest_days_a', 'rest_category': 'rest_category_a'})
          panel = panel.merge(conf_away[['game_date', 'away_team', 'conf_attendance_a',
              'avg_ref_rest_days_a', 'rest_category_a']],
              left_on=['game_date', 'team'], right_on=['game_date', 'away_team'], how='left')
          panel.drop(columns=['away_team_y'], errors='ignore', inplace=True)
          if 'away_team_x' in panel.columns: panel.rename(columns={'away_team_x': 'away_team'}, inplace=True)
          panel['conf_attendance'] = panel['conf_attendance_h'].fillna(panel['conf_attendance_a'])
          panel['avg_ref_rest_days'] = panel['avg_ref_rest_days_h'].fillna(panel['avg_ref_rest_days_a'])
          panel['rest_category'] = panel['rest_category_h'].fillna(panel['rest_category_a'])
          panel.drop(columns=['conf_attendance_h', 'conf_attendance_a',
              'avg_ref_rest_days_h', 'avg_ref_rest_days_a',
              'rest_category_h', 'rest_category_a'], inplace=True)
          panel['conf_attendance'] = panel['conf_attendance'].fillna(0)
          panel['avg_ref_rest_days'] = panel['avg_ref_rest_days'].fillna(0)
          panel['rest_category'] = panel['rest_category'].fillna('no_game')
          n_matched = (panel['game_day'] == 1).sum()
          n_with_conf = ((panel['game_day'] == 1) & (panel['conf_attendance'] > 0)).sum()
          print(f"Confounders merged: {n_with_conf:,}/{n_matched:,} game-day rows have attendance data")

      # Time components
      panel['dow'] = panel['game_date'].dt.day_name()
      panel['month'] = panel['game_date'].dt.month_name()
      panel['year'] = panel['game_date'].dt.year.astype(str)
      panel['year_num'] = panel['game_date'].dt.year
      panel['is_weekend'] = panel['game_date'].dt.dayofweek.isin([5, 6]).astype(int)
      cal = USFederalHolidayCalendar()
      holidays = set(pd.to_datetime(cal.holidays(start='2012-01-01', end='2026-01-01')))
      panel['holiday'] = panel['game_date'].isin(holidays).astype(int)

      if 'season' in panel.columns:
          season_end = (panel[(panel['game_day'] == 1) & (panel['is_playoff'] == 0)]
              .groupby('season')['game_date'].max().rename('season_end'))
          panel = panel.merge(season_end, on='season', how='left')
          panel['weeks_left'] = ((panel['season_end'] - panel['game_date']).dt.days / 7).round(1)
          is_playoff_row = (panel['game_day'] == 1) & (panel['is_playoff'] == 1)
          panel['weeks_left'] = panel['weeks_left'].fillna(0)
          panel.loc[is_playoff_row, 'weeks_left'] = -1
          panel.loc[~is_playoff_row, 'weeks_left'] = panel.loc[~is_playoff_row, 'weeks_left'].clip(lower=0)
          panel['late_season'] = ((panel['weeks_left'] > 0) & (panel['weeks_left'] <= 6)).astype(int)
          panel.drop(columns=['season_end'], inplace=True)
          print(f"weeks_left range [{panel.loc[panel['game_day']==1, 'weeks_left'].min()}, "
                f"{panel.loc[panel['game_day']==1, 'weeks_left'].max()}], late_season: {panel['late_season'].sum():,}")

      if 'time' in panel.columns and 'tipoff_hour' in panel.columns:
          missing_tipoff = ((panel['game_day'] == 1) & panel['tipoff_hour'].isna() &
              panel['time'].notna() & (panel['time'].str.strip() != ''))
          if missing_tipoff.any():
              utc_h = pd.to_datetime(panel.loc[missing_tipoff, 'time'], format='%H:%M', errors='coerce').dt.hour
              panel.loc[missing_tipoff, 'tipoff_hour'] = (utc_h - 4) % 24
              print(f"tipoff_hour fix: {missing_tipoff.sum():,} game rows")

      for col in ['national_tv', 'is_playoff', 'PACE', 'tipoff_hour']:
          if col in panel.columns: panel[col] = panel[col].fillna(0)

      if 'team' in panel.columns:
          panel['team_year'] = panel['team'] + '_' + panel['year']
          if self.league.lower() == 'nba':
              _cardazzi_teams = {'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
                  'Houston Rockets', 'Phoenix Suns', 'Milwaukee Bucks',
                  'Oklahoma City Thunder', 'Utah Jazz'}
              panel['cardazzi_team'] = panel['team'].isin(_cardazzi_teams).astype(int)

      # ── ORI quality filters (county, grey_zone, county_only, pop_cap applied pre-zero-fill) ──
      if (self.min_coverage > 0 or self.min_seasons > 0) and 'season' in panel.columns:
          n_ori_before = panel['ori'].nunique()
          n_rows_before = len(panel)

          if self.min_coverage > 0 and _cov_data is not None:
              ori_season = _cov_data
              # Drop individual ORI-seasons below threshold (not entire ORI)
              good = ori_season[ori_season['coverage'] >= self.min_coverage][['ori', 'season']]
              bad_pairs = len(ori_season) - len(good)
              # Remove rows belonging to bad ORI-season pairs
              panel['_keep'] = False
              # Non-game-day rows have NaN season — keep them if the ORI has any good season
              good_oris = good['ori'].unique()
              panel.loc[panel['season'].isna() & panel['ori'].isin(good_oris), '_keep'] = True
              # Game-day rows: keep only if ORI-season pair is good
              panel = panel.merge(good.assign(_good=True), on=['ori', 'season'], how='left')
              panel.loc[panel['_good'] == True, '_keep'] = True
              panel = panel[panel['_keep']].drop(columns=['_keep', '_good'])
              med_cov = good.merge(ori_season, on=['ori', 'season'])['coverage'].median()
              print(f"Coverage filter (>={self.min_coverage:.0%}/season): "
                    f"dropped {bad_pairs:,} bad ORI-season pairs, "
                    f"{panel['ori'].nunique():,} ORIs remain (median coverage: {med_cov:.0%})")

          if self.min_seasons > 0:
              # Count good seasons per ORI
              ori_nyears = (panel[panel['game_day'] == 1]
                  .groupby('ori')['season'].nunique())
              keep_oris = ori_nyears[ori_nyears >= self.min_seasons].index
              panel = panel[panel['ori'].isin(keep_oris)]
              print(f"Longevity filter (>={self.min_seasons} seasons): "
                    f"{panel['ori'].nunique():,} ORIs remain")

          n_ori_after = panel['ori'].nunique()
          print(f"ORI quality filters total: {n_ori_before:,} -> {n_ori_after:,} ORIs, "
                f"{n_rows_before:,} -> {len(panel):,} rows")

      panel.reset_index(drop=True, inplace=True)

      n_missing_pop = panel['population_estimate'].isna().sum()
      if n_missing_pop > 0:
          print(f"WARNING: {n_missing_pop:,} rows missing population_estimate ({n_missing_pop/len(panel)*100:.1f}%)")
      panel = panel.merge(handle, on=['state', 'year', 'month'], how='left')
      n_handle = panel['handle_per_capita'].notna().sum()
      print(f"Handle merge: {n_handle:,}/{len(panel):,} rows ({n_handle/len(panel)*100:.1f}%)")
      panel = panel.drop_duplicates()
      self._panel = panel
      return panel


    def __call__(self):
      return self.panel()


