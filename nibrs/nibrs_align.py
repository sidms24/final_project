# ── CORE PROCESSING FUNCTION ───────────────────────────

import pandas as pd
import numpy as np
import gc
ROOT = '/content/drive/MyDrive/IPV_Project'
DATA = f'{ROOT}/data'
NIBRS_RAW = '/content/drive/MyDrive/NIBRS_dta'
OUTPUT = f'{NIBRS_RAW}/processed_data_v2'

from relationships import (IPV_RELATIONS, DV_RELATIONS_ALL,OTHERFAM_VALUES, CHILD_VALUES,
                           KNOWN_VALUES,SERIOUS_INJURIES)


def process_nibrs_year_v2(year, data_path=NIBRS_RAW):
    """
    Corrected NIBRS processing for one year.

    Key differences from v1:
    - Merges victim×offender WITHIN incident first, then checks relationship
    - Deduplicates to ONE row per incident (FIX #1)
    - Excludes 'other - known to victim' from otherfam (FIX #2)
    - Filters for male-offender × female-victim IPV (FIX #3)
    """
    print(f"\n{'='*60}")
    print(f"Processing year: {year}")
    print(f"{'='*60}")

    try:
        # ── STEP 1: Load offense → filter for assault ──
        print(f"  [1/7] Loading offense segment...")
        offense_df = pd.read_csv(
            f'{data_path}/offense_segment_csv_1991_2024/nibrs_offense_segment_{year}.csv',
            low_memory=False
        )
        assault_mask = offense_df['ucr_offense_code'].str.contains(
            'assault offenses', case=False, na=False
        )
        offense_assault = offense_df[assault_mask].copy()
        assault_incidents = set(offense_assault['incident_number'].unique())
        print(f"       {len(assault_incidents):,} unique assault incidents")

        # Keep only needed offense columns
        offense_keep = ['ori', 'incident_number', 'ucr_offense_code',
                        'offender_suspected_of_using_1', 'offender_suspected_of_using_2',
                        'offender_suspected_of_using_3', 'location_type']
        offense_keep = [c for c in offense_keep if c in offense_assault.columns]
        offense_assault = offense_assault[offense_keep]
        del offense_df; gc.collect()

        # ── STEP 2: Load admin → filter to assault incidents ──
        print(f"  [2/7] Loading admin segment...")
        admin_df = pd.read_csv(
            f'{data_path}/administrative_segment_csv_1991_2024/nibrs_administrative_segment_{year}.csv',
            low_memory=False
        )
        admin_df = admin_df[admin_df['incident_number'].isin(assault_incidents)].copy()
        admin_keep = ['ori', 'year', 'state', 'state_abb', 'incident_number',
                      'incident_date', 'incident_date_hour', 'city_submissions']
        admin_keep = [c for c in admin_keep if c in admin_df.columns]
        admin_df = admin_df[admin_keep]
        print(f"       {len(admin_df):,} admin records")

        # ── STEP 3: Load victim → filter to assault, process relationships ──
        print(f"  [3/7] Loading & processing victim segment...")
        victim_df = pd.read_csv(
            f'{data_path}/victim_segment_csv_1991_2024/nibrs_victim_segment_{year}.csv',
            low_memory=False
        )
        victim_df = victim_df[victim_df['incident_number'].isin(assault_incidents)].copy()

        # Process victim demographics
        victim_df['vage'] = pd.to_numeric(victim_df['age_of_victim'], errors='coerce')
        victim_df.loc[victim_df['vage'] == 0, 'vage'] = np.nan

        sex_clean = victim_df['sex_of_victim'].str.lower().str.strip()
        victim_df['vfemale'] = (sex_clean == 'female').astype(float)
        victim_df.loc[sex_clean == 'unknown', 'vfemale'] = np.nan

        # Process ALL relationship columns
        rel_cols = [f'relation_of_vict_to_offender{i}' for i in range(1, 11)]
        rel_cols = [c for c in rel_cols if c in victim_df.columns]
        rel_lower = victim_df[rel_cols].fillna('').astype(str).apply(
            lambda x: x.str.lower().str.strip()
        )

        # Relationship flags
        victim_df['spouse'] = (rel_lower == 'victim was spouse').any(axis=1).astype(int)
        victim_df['commonspouse'] = (rel_lower == 'victim was common-law spouse').any(axis=1).astype(int)
        victim_df['exspouse'] = (rel_lower == 'victim was ex-spouse').any(axis=1).astype(int)
        victim_df['bgfriend'] = (rel_lower == 'victim was boyfriend/girlfriend').any(axis=1).astype(int)
        victim_df['child'] = rel_lower.isin(CHILD_VALUES).any(axis=1).astype(int)
        victim_df['otherfam'] = rel_lower.isin(OTHERFAM_VALUES).any(axis=1).astype(int)  # FIX #2
        victim_df['known'] = rel_lower.isin(KNOWN_VALUES).any(axis=1).astype(int)
        victim_df['stranger'] = (rel_lower == 'victim was stranger').any(axis=1).astype(int)

        # Composites
        victim_df['intpartner'] = (
            (victim_df['spouse'] == 1) | (victim_df['commonspouse'] == 1) |
            (victim_df['exspouse'] == 1) | (victim_df['bgfriend'] == 1)
        ).astype(int)
        victim_df['anyspouse'] = (
            (victim_df['spouse'] == 1) | (victim_df['commonspouse'] == 1) |
            (victim_df['exspouse'] == 1)
        ).astype(int)
        victim_df['extfamily'] = ((victim_df['child'] == 1) | (victim_df['otherfam'] == 1)).astype(int)

        # Injury processing
        injury_cols = [f'type_of_injury_{i}' for i in range(1, 6)]
        injury_cols = [c for c in injury_cols if c in victim_df.columns]
        if injury_cols:
            inj_lower = victim_df[injury_cols].fillna('').astype(str).apply(
                lambda x: x.str.lower().str.strip()
            )
            # FIX: injuryN should check ALL injury cols are 'none' or empty
            victim_df['injuryN'] = (
                (inj_lower == 'none') | (inj_lower == '')
            ).all(axis=1).astype(int)
            victim_df['injuryS'] = inj_lower.isin(SERIOUS_INJURIES).any(axis=1).astype(int)
            victim_df['injuryM'] = (
                (inj_lower == 'apparent minor injury').any(axis=1) &
                (victim_df['injuryS'] == 0)
            ).astype(int)

        # Filter: only DV-related victims (any relationship col matches)
        dv_mask = rel_lower.isin([v.lower() for v in DV_RELATIONS_ALL]).any(axis=1)
        victim_dv = victim_df[dv_mask].copy()
        print(f"       {len(victim_dv):,} DV victims (from {len(victim_df):,} assault victims)")
        del victim_df, rel_lower; gc.collect()

        # ── STEP 4: Load offender → filter to assault, process ──
        print(f"  [4/7] Loading & processing offender segment...")
        offender_df = pd.read_csv(
            f'{data_path}/offender_segment_csv_1991_2024/nibrs_offender_segment_{year}.csv',
            low_memory=False
        )
        offender_df = offender_df[offender_df['incident_number'].isin(assault_incidents)].copy()

        offender_df['oage'] = pd.to_numeric(offender_df['age_of_offender'], errors='coerce')
        offender_df.loc[offender_df['oage'] == 0, 'oage'] = np.nan
        osex = offender_df['sex_of_offender'].str.lower().str.strip()
        offender_df['ofemale'] = (osex == 'female').astype(float)
        offender_df.loc[osex == 'unknown', 'ofemale'] = np.nan

        off_keep = ['ori', 'incident_number', 'offender_sequence_number',
                     'oage', 'ofemale']
        off_keep = [c for c in off_keep if c in offender_df.columns]
        offender_df = offender_df[off_keep]
        print(f"       {len(offender_df):,} offender records")

        # ── STEP 5: Merge victim × offender on incident ──
        print(f"  [5/7] Merging victim × offender...")
        vo = victim_dv.merge(
            offender_df,
            on=['ori', 'incident_number'],
            how='inner'
        )
        print(f"       {len(vo):,} victim-offender pairs (before gender filter)")
        del victim_dv, offender_df; gc.collect()

        # ── FIX #3: Filter for male-offender × female-victim ──
        print(f"  [6/7] Applying male-offender × female-victim filter...")
        n_before = len(vo)
        vo = vo[(vo['vfemale'] == 1) & (vo['ofemale'] == 0)].copy()
        print(f"       {len(vo):,} male→female pairs (dropped {n_before - len(vo):,})")

        # ── FIX #1: Deduplicate to ONE ROW PER INCIDENT ──
        # An incident should count as 1 IPV event regardless of how many
        # victim-offender pairs it contains
        print(f"  [7/7] Deduplicating to one row per incident...")
        n_before_dedup = len(vo)

        # For each incident, take the MAX of relationship flags
        # (if ANY pair is intpartner, the incident is intpartner)
        rel_flags = ['spouse', 'commonspouse', 'exspouse', 'bgfriend',
                     'child', 'otherfam', 'known', 'stranger',
                     'intpartner', 'anyspouse', 'extfamily',
                     'injuryN', 'injuryM', 'injuryS']
        rel_flags = [c for c in rel_flags if c in vo.columns]

        agg_dict = {col: 'max' for col in rel_flags}
        # For demographics, take first (arbitrary but consistent)
        for col in ['vage', 'vfemale', 'oage', 'ofemale']:
            if col in vo.columns:
                agg_dict[col] = 'first'

        deduped = vo.groupby(
            ['ori', 'incident_number'], as_index=False
        ).agg(agg_dict)
        print(f"       {len(deduped):,} unique incidents (from {n_before_dedup:,} pairs)")
        del vo; gc.collect()

        # ── Merge with admin for date/state info ──
        merged = deduped.merge(admin_df, on=['ori', 'incident_number'], how='inner')
        print(f"       {len(merged):,} final records after admin merge")
        del deduped, admin_df; gc.collect()

        # ── Merge offense info (take first offense per incident) ──
        offense_dedup = offense_assault.groupby(
            ['ori', 'incident_number'], as_index=False
        ).first()
        merged = merged.merge(offense_dedup, on=['ori', 'incident_number'], how='inner')
        del offense_assault, offense_dedup; gc.collect()

        print(f"\n✓ Year {year}: {len(merged):,} clean IPV incidents")
        return merged

    except Exception as e:
        print(f"\n✗ Error processing year {year}: {e}")
        import traceback
        traceback.print_exc()
        return None
