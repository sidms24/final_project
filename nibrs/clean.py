from nibrs_align import process_nibrs_year_v2, OUTPUT, DATA
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import gc 

if __name__ == "__main__":
    all_results = []
    failed_years = []
    for year in range(2011, 2025):
        df_year = process_nibrs_year_v2(year)
        if df_year is not None and len(df_year) > 0:
            # Save individual year
            df_year.to_csv(f'{OUTPUT}/dv_data_{year}_v2.csv', index=False)
            print(f"  → Saved to {OUTPUT}/dv_data_{year}_v2.csv")
            all_results.append(df_year)
            del df_year; gc.collect()
        else:
            failed_years.append(year)

    combined = pd.concat(all_results, ignore_index=True)
