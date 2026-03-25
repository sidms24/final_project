import pandas as pd
def filter_consistent_reporters(df, min_years=5):
    if 'year' in df.columns:
        years = df['year'].astype(str).str[:4]
    else:
        years = df['game_date'].dt.year.astype(str)
    ori_year_counts = years.groupby(df['ori']).nunique()
    consistent_oris = set(ori_year_counts[ori_year_counts >= min_years].index)
    print(f"filter_consistent_reporters: {len(consistent_oris)}/{df['ori'].nunique()} "
          f"ORIs kept (>={min_years} years)")
    return consistent_oris


def load_game_day_reporting(league):
    url = (f'https://huggingface.co/datasets/group-a/Final_Project/resolve/main/'
           f'cleaned_nibrs/ori_game_day_reporting_{league.lower()}.parquet')
    df = pd.read_parquet(url)
    n_pass = df['passes'].sum()
    print(f"Game-day reporting ({league.upper()}): {n_pass:,}/{len(df):,} ORI-seasons pass, "
          f"{df.loc[df['passes'], 'ori'].nunique():,} unique ORIs")
    return df


def get_passing_ori_seasons(reporting_df, ori_candidates):
    df = reporting_df[
        (reporting_df['ori'].isin(ori_candidates)) &
        (reporting_df['passes'] == True)
    ]
    passing_pairs = set(zip(df['ori'], df['season']))
    passing_oris = set(df['ori'])
    print(f"get_passing_ori_seasons: {len(passing_oris):,}/{len(ori_candidates):,} ORIs "
          f"have >=1 passing season ({len(passing_pairs):,} ORI-season pairs)")
    return passing_pairs, passing_oris