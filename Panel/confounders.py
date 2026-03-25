import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

def load_game_controls(league):
    if league.lower() == 'nba':
      bc = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/Controls/ESPN_nba_broadcasters_2009_2026.csv')
      pace = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/Controls/NBA_game_minutes_pace_2009_2025_v3.csv')
    elif league.lower() == 'wnba':
      bc = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/Controls/ESPN_wnba_broadcasters_2009_2025.csv')
      pace = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/Controls/WNBA_game_minutes_pace_2009_2025_v3.csv')
    bc['game_date_time'] = pd.to_datetime(bc['game_date_time'], format='mixed', utc=True)
    bc['game_date'] = bc['game_date_time'].dt.tz_localize(None).dt.normalize()
    bc['tipoff_hour'] = (bc['game_date_time'].dt.tz_localize(None).dt.hour
                         + bc['game_date_time'].dt.tz_localize(None).dt.minute / 60)
    bc.rename(columns={
        'home_team_full_name': 'home_team', 'away_team_full_name': 'away_team',
        'home_score': 'home_team_score', 'away_score': 'away_team_score',
    }, inplace=True)
    bc.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], inplace=True)
    bc['national_tv'] = bc['broadcast_market'].eq('national').astype(int)
    bc = bc[['game_date', 'home_team', 'away_team',
             'home_team_score', 'away_team_score',
             'national_tv', 'tipoff_hour']].dropna()
    pace = pace[pace['season_type_description'].isin(['Regular Season', 'Playoffs'])].copy()
    pace['game_date_time_est'] = pd.to_datetime(pace['game_date_time_est'], format='mixed', utc=True)
    pace['game_date'] = pace['game_date_time_est'].dt.tz_localize(None).dt.normalize()
    pace['home_team'] = pace['home_team_city'] + ' ' + pace['home_team_name']
    pace['away_team'] = pace['away_team_city'] + ' ' + pace['away_team_name']
    pace['season'] = pace['season'].astype(str)
    pace['is_playoff'] = (pace['season_type_description'] == 'Playoffs').astype(int)
    pace = pace[['game_date', 'home_team', 'away_team',
                 'home_team_score', 'away_team_score',
                 'season', 'is_playoff', 'PACE', 'minutes']].dropna().drop_duplicates()
    games = pd.merge(pace, bc,
        on=['game_date', 'home_team', 'away_team', 'home_team_score', 'away_team_score'], how='left')
    games = games[games['game_date'] < pd.Timestamp('2025-01-01')]
    games.drop_duplicates(inplace=True)
    games.reset_index(drop=True, inplace=True)
    cal = USFederalHolidayCalendar()
    holidays = set(pd.to_datetime(cal.holidays(start='2009-01-01', end='2025-01-01')))
    games['holiday'] = games['game_date'].isin(holidays).astype(int)
    return games




def load_confounders(league):
    HF_BASE = 'https://huggingface.co/datasets/group-a/Final_Project/resolve/main'
    if league.lower() == 'nba':
        df = pd.read_csv(f'{HF_BASE}/Controls/nba_part1_confounders.csv')
        df.rename(columns={'date': 'game_date'}, inplace=True)
        df['game_date'] = pd.to_datetime(df['game_date'])
        _nba_name_map = {
            'Atlanta': 'Atlanta Hawks', 'Boston': 'Boston Celtics',
            'Brooklyn': 'Brooklyn Nets', 'Charlotte': 'Charlotte Hornets',
            'Chicago': 'Chicago Bulls', 'Cleveland': 'Cleveland Cavaliers',
            'Dallas': 'Dallas Mavericks', 'Denver': 'Denver Nuggets',
            'Detroit': 'Detroit Pistons', 'Golden State': 'Golden State Warriors',
            'Houston': 'Houston Rockets', 'Indiana': 'Indiana Pacers',
            'L.A. Lakers': 'Los Angeles Lakers', 'L.A. Clippers': 'Los Angeles Clippers',
            'Memphis': 'Memphis Grizzlies', 'Miami': 'Miami Heat',
            'Milwaukee': 'Milwaukee Bucks', 'Minnesota': 'Minnesota Timberwolves',
            'New Jersey': 'Brooklyn Nets', 'New Orleans': 'New Orleans Pelicans',
            'New York': 'New York Knicks', 'Oklahoma City': 'Oklahoma City Thunder',
            'Orlando': 'Orlando Magic', 'Philadelphia': 'Philadelphia 76ers',
            'Phoenix': 'Phoenix Suns', 'Portland': 'Portland Trail Blazers',
            'Sacramento': 'Sacramento Kings', 'San Antonio': 'San Antonio Spurs',
            'Toronto': 'Toronto Raptors', 'Utah': 'Utah Jazz',
            'Washington': 'Washington Wizards',
        }
        df['home_team'] = df['home_team'].map(_nba_name_map)
        df['away_team'] = df['away_team'].map(_nba_name_map)
        df = df.sort_values('avg_ref_rest_days', ascending=False, na_position='last')
        df = df.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='first')
        df.drop(columns=['arenaId', 'holiday_game_day'], errors='ignore', inplace=True)
    elif league.lower() == 'wnba':
        df = pd.read_csv(f'{HF_BASE}/Controls/wnba_part1_confounders.csv')
        df.rename(columns={'home_team_clean': 'home_team', 'away_team_clean': 'away_team'}, inplace=True)
        df['game_date'] = pd.to_datetime(df['game_date'])
        _wnba_hist_map = {'San Antonio Stars': 'Las Vegas Aces', 'Tulsa Shock': 'Dallas Wings'}
        df['home_team'] = df['home_team'].replace(_wnba_hist_map)
        df['away_team'] = df['away_team'].replace(_wnba_hist_map)
        df = df.dropna(subset=['away_team'])
        df.drop(columns=['holiday_game_day'], errors='ignore', inplace=True)
    df.rename(columns={'attendance': 'conf_attendance'}, inplace=True)
    keep = ['game_date', 'home_team', 'away_team', 'conf_attendance', 'avg_ref_rest_days', 'rest_category']
    df = df[[c for c in keep if c in df.columns]]
    print(f"Confounders ({league.upper()}): {len(df):,} rows, "
          f"{df['home_team'].nunique()} home teams, "
          f"attendance NaN: {df['conf_attendance'].isna().sum()}, "
          f"ref_rest NaN: {df['avg_ref_rest_days'].isna().sum()}")
    return df




def load_handle():
  handle = pd.read_csv('https://huggingface.co/datasets/group-a/Final_Project/resolve/main/Controls/betting_handle.csv')
  monthly_median = (
    handle
    .groupby(["year", "month"])["handle_per_capita"]
    .transform("median")
)

  handle["high_handle"] = (handle["handle_per_capita"] >=
  monthly_median).astype(int)
  handle["month"] = pd.to_datetime(handle["month"],format="%m").dt.month_name()
  handle["year"] = pd.to_datetime(handle["year"],format="%Y").dt.year.astype(str)

  handle = handle.rename(columns={"State":"state"})
  handle_cols = ["state", "year", "month", "handle_per_capita", "log_handle_pc", "high_handle"]
  return handle[handle_cols]


