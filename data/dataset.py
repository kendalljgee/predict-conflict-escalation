"""
COMPLETE CONFLICT ESCALATION PREDICTION PIPELINE
Full implementation with negative case generation, data merging, and standardization

This pipeline:
1. Loads all 5 COW datasets
2. Processes MIDB into dyadic disputes
3. Generates negative cases (non-dispute dyad-years)
4. Merges all datasets (capabilities, alliances, trade, contiguity)
5. Engineers features
6. Standardizes numerical features for modeling
7. Saves ready-to-use dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CONFLICT ESCALATION DATASET PIPELINE")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '/'
OUTPUT_FILE = '../conflict_escalation_dataset.csv'
NEGATIVE_SAMPLE_RATE = 0.1  # Sample 10% of non-dispute dyad-years
START_YEAR = 1816
END_YEAR = 2014
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING DATASETS")
print("=" * 80)

# Load MIDB 5.0 - Militarized Interstate Disputes (participant-level)
print("\nLoading MIDB 5.0...")
midb = pd.read_csv(f'{DATA_DIR}MIDB_5.0.csv')
print(f"  Loaded {len(midb):,} participant records")
print(f"  Columns: {list(midb.columns)}")

# Load NMC 6.0 - National Material Capabilities
print("\nLoading NMC 6.0...")
nmc = pd.read_csv(f'{DATA_DIR}NMC_6.0_abridged.csv')
print(f"  Loaded {len(nmc):,} country-year records")
print(f"  Columns: {list(nmc.columns)}")

# Load Alliance data (dyadic yearly)
print("\nLoading Alliance data...")
alliance = pd.read_csv(f'{DATA_DIR}alliance_v4.1_by_dyad_yearly.csv')
print(f"  Loaded {len(alliance):,} dyad-year records")
print(f"  Columns: {list(alliance.columns)}")

# Load Dyadic Trade 4.0
print("\nLoading Trade data...")
trade = pd.read_csv(f'{DATA_DIR}Dyadic_COW_4.0.csv')
print(f"  Loaded {len(trade):,} dyad-year records")
print(f"  Columns: {list(trade.columns)}")

# Load Direct Contiguity
print("\nLoading Contiguity data...")
contiguity = pd.read_csv(f'{DATA_DIR}contdird.csv')
print(f"  Loaded {len(contiguity):,} dyad-year records")
print(f"  Columns: {list(contiguity.columns)}")

print("\n✓ All datasets loaded successfully!")

# ============================================================================
# STEP 2: PROCESS MIDB INTO DYADIC DISPUTES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: PROCESSING MIDB INTO DYADIC FORMAT")
print("=" * 80)

print("\nMIDB is participant-level. Converting to dyadic (country-pair) format...")

# Separate Side A and Side B participants
sidea = midb[midb['sidea'] == 1].copy()
sideb = midb[midb['sidea'] == 0].copy()

print(f"  Side A participants: {len(sidea):,}")
print(f"  Side B participants: {len(sideb):,}")

# Create dyads by matching participants within same dispute
print("\nCreating dyads from disputes...")
dyads = []

for dispnum in midb['dispnum'].unique():
    # Get all participants on each side for this dispute
    dispute_a = sidea[sidea['dispnum'] == dispnum]
    dispute_b = sideb[sideb['dispnum'] == dispnum]

    if len(dispute_a) == 0 or len(dispute_b) == 0:
        continue

    # Get dispute-level information
    year = dispute_a['styear'].iloc[0]
    max_hostlev = midb[midb['dispnum'] == dispnum]['hostlev'].max()
    total_fatality = midb[midb['dispnum'] == dispnum]['fatality'].sum()

    # Create all possible dyads between Side A and Side B
    for _, a in dispute_a.iterrows():
        for _, b in dispute_b.iterrows():
            dyads.append({
                'dispnum': dispnum,
                'statea': a['ccode'],
                'stateb': b['ccode'],
                'year': year,
                'hostlev': max_hostlev,
                'fatality': total_fatality,
                'hiact_a': a['hiact'],
                'hiact_b': b['hiact'],
                'orig_a': a['orig'],
                'orig_b': b['orig']
            })

midb_dyadic = pd.DataFrame(dyads)
print(f"  Created {len(midb_dyadic):,} dyadic records")

# Ensure consistent dyad ordering: statea < stateb
print("\nStandardizing dyad ordering (statea < stateb)...")
mask = midb_dyadic['statea'] > midb_dyadic['stateb']
midb_dyadic.loc[mask, ['statea', 'stateb']] = midb_dyadic.loc[mask, ['stateb', 'statea']].values

# Remove duplicate dyads within same dispute
midb_dyadic = midb_dyadic.drop_duplicates(subset=['dispnum', 'statea', 'stateb'])
print(f"  After removing duplicates: {len(midb_dyadic):,} dyads")

# CREATE TARGET VARIABLE: escalation = 1 if hostlev >= 4
# hostlev: 1=no action, 2=threat, 3=display, 4=use of force, 5=war
midb_dyadic['escalation'] = (midb_dyadic['hostlev'] >= 4).astype(int)

print("\n" + "-" * 80)
print("DISPUTE SUMMARY:")
print("-" * 80)
print(f"Total dyadic disputes: {len(midb_dyadic):,}")
print(
    f"Escalated disputes (hostlev >= 4): {midb_dyadic['escalation'].sum():,} ({midb_dyadic['escalation'].mean() * 100:.1f}%)")
print(
    f"Non-escalated disputes (hostlev < 4): {(midb_dyadic['escalation'] == 0).sum():,} ({(1 - midb_dyadic['escalation'].mean()) * 100:.1f}%)")
print(f"Year range: {midb_dyadic['year'].min():.0f} - {midb_dyadic['year'].max():.0f}")
print("-" * 80)

# ============================================================================
# STEP 3: GENERATE NEGATIVE CASES (NON-DISPUTE DYAD-YEARS)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: GENERATING NEGATIVE CASES")
print("=" * 80)

print("\nNegative cases = dyad-years where NO dispute occurred")
print(f"Sampling rate: {NEGATIVE_SAMPLE_RATE * 100}% of all possible non-dispute dyad-years")
print(f"Year range: {START_YEAR} - {END_YEAR}")

# Get all country-years from NMC (which countries existed in which years)
countries = nmc[['ccode', 'year']].drop_duplicates()
countries = countries[(countries['year'] >= START_YEAR) &
                      (countries['year'] <= END_YEAR)]

print(f"\nCountries in NMC data: {countries['ccode'].nunique():,} unique countries")
print(f"Country-years: {len(countries):,}")

# Create set of dispute dyad-years to exclude
print("\nBuilding set of dispute dyad-years to exclude...")
dispute_dyad_years = set()
for _, row in midb_dyadic.iterrows():
    a, b = sorted([row['statea'], row['stateb']])
    dispute_dyad_years.add((a, b, row['year']))

print(f"  Dispute dyad-years to exclude: {len(dispute_dyad_years):,}")

# Generate negative cases
print("\nGenerating non-dispute dyad-years...")
negative_cases = []
years = sorted(countries['year'].unique())

for idx, year in enumerate(years):
    if idx % 20 == 0:
        print(f"  Processing year {year}... ({idx + 1}/{len(years)})")

    # Get all countries that existed in this year
    year_countries = sorted(countries[countries['year'] == year]['ccode'].unique())

    # Generate all possible dyads
    all_dyads = list(combinations(year_countries, 2))

    # Sample dyads for computational efficiency
    n_sample = max(1, int(len(all_dyads) * NEGATIVE_SAMPLE_RATE))
    sampled_indices = np.random.choice(len(all_dyads), size=n_sample, replace=False)

    for idx in sampled_indices:
        statea, stateb = all_dyads[idx]

        # Ensure ordering
        if statea > stateb:
            statea, stateb = stateb, statea

        # Skip if this dyad-year had a dispute
        if (statea, stateb, year) in dispute_dyad_years:
            continue

        negative_cases.append({
            'statea': statea,
            'stateb': stateb,
            'year': year,
            'escalation': 0,
            'hostlev': 0,
            'fatality': 0
        })

negative_df = pd.DataFrame(negative_cases)

print("\n" + "-" * 80)
print("NEGATIVE CASE SUMMARY:")
print("-" * 80)
print(f"Generated negative cases: {len(negative_df):,}")
print(f"Estimated total non-dispute dyad-years: ~{len(negative_df) / NEGATIVE_SAMPLE_RATE:,.0f}")
print(f"Sample rate: {NEGATIVE_SAMPLE_RATE * 100}%")
print("-" * 80)

# Combine positive (disputes) and negative (non-disputes) cases
print("\nCombining positive and negative cases...")
df_combined = pd.concat([midb_dyadic, negative_df], ignore_index=True)

print("\n" + "-" * 80)
print("COMBINED DATASET:")
print("-" * 80)
print(f"Total observations: {len(df_combined):,}")
print(
    f"  Positive cases (escalation=1): {df_combined['escalation'].sum():,} ({df_combined['escalation'].mean() * 100:.2f}%)")
print(
    f"  Negative cases (escalation=0): {(df_combined['escalation'] == 0).sum():,} ({(1 - df_combined['escalation'].mean()) * 100:.2f}%)")
print(
    f"  Imbalance ratio: {(df_combined['escalation'] == 0).sum() / df_combined['escalation'].sum():.1f}:1 (negative:positive)")
print("-" * 80)

# ============================================================================
# STEP 4: MERGE NATIONAL MATERIAL CAPABILITIES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: MERGING NATIONAL MATERIAL CAPABILITIES")
print("=" * 80)

print("\nMerging NMC data for State A and State B...")

# Select relevant NMC columns
nmc_cols = ['ccode', 'year', 'irst', 'milex', 'milper', 'pec', 'tpop', 'upop', 'cinc']
nmc_subset = nmc[nmc_cols].copy()

initial_size = len(df_combined)

# Merge for State A
print("  Merging State A capabilities...")
df_merged = df_combined.merge(
    nmc_subset,
    left_on=['statea', 'year'],
    right_on=['ccode', 'year'],
    how='left',
    suffixes=('', '_temp')
)
df_merged = df_merged.drop('ccode', axis=1)

# Rename columns for State A
for col in ['irst', 'milex', 'milper', 'pec', 'tpop', 'upop', 'cinc']:
    if col in df_merged.columns:
        df_merged = df_merged.rename(columns={col: f'{col}_a'})

# Merge for State B
print("  Merging State B capabilities...")
df_merged = df_merged.merge(
    nmc_subset,
    left_on=['stateb', 'year'],
    right_on=['ccode', 'year'],
    how='left',
    suffixes=('', '_temp')
)
df_merged = df_merged.drop('ccode', axis=1)

# Rename columns for State B
for col in ['irst', 'milex', 'milper', 'pec', 'tpop', 'upop', 'cinc']:
    if col in df_merged.columns:
        df_merged = df_merged.rename(columns={col: f'{col}_b'})

# Create derived capability features
print("\n  Creating derived capability features...")
df_merged['cinc_ratio'] = df_merged['cinc_a'] / (df_merged['cinc_b'] + 0.0001)
df_merged['cinc_diff'] = df_merged['cinc_a'] - df_merged['cinc_b']
df_merged['power_parity'] = 1 - abs(df_merged['cinc_a'] - df_merged['cinc_b']) / (
        df_merged['cinc_a'] + df_merged['cinc_b'] + 0.0001)
df_merged['milex_ratio'] = df_merged['milex_a'] / (df_merged['milex_b'] + 0.0001)
df_merged['tpop_ratio'] = df_merged['tpop_a'] / (df_merged['tpop_b'] + 0.0001)
df_merged['joint_cinc'] = df_merged['cinc_a'] + df_merged['cinc_b']

n_merged = df_merged[['cinc_a', 'cinc_b']].notna().all(axis=1).sum()
print(
    f"\n  Successfully merged capabilities for {n_merged:,}/{len(df_merged):,} observations ({n_merged / len(df_merged) * 100:.1f}%)")

# ============================================================================
# STEP 5: MERGE ALLIANCE DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: MERGING ALLIANCE DATA")
print("=" * 80)

print("\nMerging alliance data...")

# Ensure consistent dyad ordering in alliance data
alliance['ccode1_ord'] = alliance[['ccode1', 'ccode2']].min(axis=1)
alliance['ccode2_ord'] = alliance[['ccode1', 'ccode2']].max(axis=1)

# Select alliance columns
alliance_cols = ['ccode1_ord', 'ccode2_ord', 'year', 'defense',
                 'neutrality', 'nonaggression', 'entente']
alliance_subset = alliance[alliance_cols].copy()

# Merge
df_merged = df_merged.merge(
    alliance_subset,
    left_on=['statea', 'stateb', 'year'],
    right_on=['ccode1_ord', 'ccode2_ord', 'year'],
    how='left'
)

# Clean up
df_merged = df_merged.drop(['ccode1_ord', 'ccode2_ord'], axis=1, errors='ignore')

# Fill NaN with 0 (no alliance)
alliance_types = ['defense', 'neutrality', 'nonaggression', 'entente']
for col in alliance_types:
    df_merged[col] = df_merged[col].fillna(0)

# Create any_alliance indicator
df_merged['any_alliance'] = df_merged[alliance_types].max(axis=1)

n_alliances = df_merged['any_alliance'].sum()
print(f"  Found alliances in {n_alliances:,}/{len(df_merged):,} dyad-years ({n_alliances / len(df_merged) * 100:.1f}%)")

# ============================================================================
# STEP 6: MERGE TRADE DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: MERGING TRADE DATA")
print("=" * 80)

print("\nMerging bilateral trade data...")

# Ensure consistent dyad ordering in trade data
trade['ccode1_ord'] = trade[['ccode1', 'ccode2']].min(axis=1)
trade['ccode2_ord'] = trade[['ccode1', 'ccode2']].max(axis=1)

# Select relevant columns (use smoothed flows - more complete)
trade_cols = ['ccode1_ord', 'ccode2_ord', 'year', 'smoothflow1',
              'smoothflow2', 'smoothtotrade']
trade_subset = trade[trade_cols].copy()

# Merge
df_merged = df_merged.merge(
    trade_subset,
    left_on=['statea', 'stateb', 'year'],
    right_on=['ccode1_ord', 'ccode2_ord', 'year'],
    how='left'
)

# Clean up
df_merged = df_merged.drop(['ccode1_ord', 'ccode2_ord'], axis=1, errors='ignore')

# Use smoothtotrade if available, otherwise sum flows
if 'smoothtotrade' in df_merged.columns:
    df_merged['total_trade'] = df_merged['smoothtotrade']
else:
    df_merged['total_trade'] = df_merged['smoothflow1'].fillna(0) + df_merged['smoothflow2'].fillna(0)

n_trade = df_merged['total_trade'].notna().sum()
print(f"  Found trade data for {n_trade:,}/{len(df_merged):,} dyad-years ({n_trade / len(df_merged) * 100:.1f}%)")

# ============================================================================
# STEP 7: MERGE CONTIGUITY DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: MERGING CONTIGUITY DATA")
print("=" * 80)

print("\nMerging direct contiguity data...")

# Ensure consistent dyad ordering in contiguity data
contiguity['state1_ord'] = contiguity[['state1no', 'state2no']].min(axis=1)
contiguity['state2_ord'] = contiguity[['state1no', 'state2no']].max(axis=1)

# Select columns
cont_cols = ['state1_ord', 'state2_ord', 'year', 'conttype']
contiguity_subset = contiguity[cont_cols].copy()

# Merge
df_merged = df_merged.merge(
    contiguity_subset,
    left_on=['statea', 'stateb', 'year'],
    right_on=['state1_ord', 'state2_ord', 'year'],
    how='left'
)

# Clean up
df_merged = df_merged.drop(['state1_ord', 'state2_ord'], axis=1, errors='ignore')

# Create contiguity indicators
# conttype: 1=land, 2=sea, 3=river, 4=lake, 5=12-24 miles, 6=24-150 miles
df_merged['is_contiguous'] = df_merged['conttype'].notna().astype(int)
df_merged['land_contiguous'] = (df_merged['conttype'] == 1).astype(int)
df_merged['sea_contiguous'] = (df_merged['conttype'] == 2).astype(int)

# Fill conttype NaN with 0 (not contiguous)
df_merged['conttype'] = df_merged['conttype'].fillna(0)

n_contiguous = df_merged['is_contiguous'].sum()
print(
    f"  Found contiguity in {n_contiguous:,}/{len(df_merged):,} dyad-years ({n_contiguous / len(df_merged) * 100:.1f}%)")
print(f"    Land contiguous: {df_merged['land_contiguous'].sum():,}")
print(f"    Sea contiguous: {df_merged['sea_contiguous'].sum():,}")

# ============================================================================
# STEP 8: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: ENGINEERING ADDITIONAL FEATURES")
print("=" * 80)

print("\nCreating derived features...")

# Economic interdependence (trade normalized by population)
df_merged['trade_per_capita_a'] = df_merged['total_trade'] / (df_merged['tpop_a'] * 1000 + 0.0001)
df_merged['trade_per_capita_b'] = df_merged['total_trade'] / (df_merged['tpop_b'] * 1000 + 0.0001)

# Urbanization ratios
df_merged['urban_ratio_a'] = df_merged['upop_a'] / (df_merged['tpop_a'] + 0.0001)
df_merged['urban_ratio_b'] = df_merged['upop_b'] / (df_merged['tpop_b'] + 0.0001)

# Industrial capacity ratio
df_merged['irst_ratio'] = df_merged['irst_a'] / (df_merged['irst_b'] + 0.0001)

# Geographic distance proxy
df_merged['geographic_distance'] = 1 - df_merged['is_contiguous']

print("  Created derived features:")
print("    - Trade per capita (A & B)")
print("    - Urbanization ratios (A & B)")
print("    - Industrial capacity ratio")
print("    - Geographic distance proxy")

# ============================================================================
# STEP 9: HANDLE MISSING VALUES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: HANDLING MISSING VALUES")
print("=" * 80)

print("\nChecking for missing values...")
missing = df_merged.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print(f"\nColumns with missing values:")
    print(missing.head(20))

    print("\nFilling missing values with 0 (will represent mean after standardization)...")
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)
    print("  ✓ Missing values filled")
else:
    print("  ✓ No missing values found")

# Remove rows with missing key identifiers
print("\nRemoving rows with missing key identifiers...")
initial_len = len(df_merged)
df_merged = df_merged[df_merged['statea'].notna() &
                      df_merged['stateb'].notna() &
                      df_merged['year'].notna()].copy()
print(f"  Removed {initial_len - len(df_merged):,} rows with missing identifiers")

# ============================================================================
# STEP 10: STANDARDIZE NUMERICAL FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: STANDARDIZING NUMERICAL FEATURES")
print("=" * 80)

print("\nIdentifying features to standardize...")

# Columns to exclude from standardization
exclude_cols = [
    'escalation',  # Target variable
    'statea', 'stateb', 'year', 'dispnum',  # Identifiers
    'hostlev', 'fatality',  # Keep original scale for interpretation
    'is_contiguous', 'land_contiguous', 'sea_contiguous',  # Binary indicators
    'defense', 'neutrality', 'nonaggression', 'entente', 'any_alliance',  # Binary
    'hiact_a', 'hiact_b', 'orig_a', 'orig_b',  # Categorical
    'conttype'  # Categorical
]

# Identify numerical columns
numerical_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()

# Remove excluded columns
cols_to_standardize = [c for c in numerical_cols if c not in exclude_cols]

print(f"\nFeatures to standardize: {len(cols_to_standardize)}")
print("\nStandardizing features (mean=0, std=1)...")

# Create standardizer
scaler = StandardScaler()

# Create standardized dataset
df_standardized = df_merged.copy()
df_standardized[cols_to_standardize] = scaler.fit_transform(
    df_merged[cols_to_standardize]
)

print("  ✓ Standardization complete")

print("\nStandardized features:")
for i, col in enumerate(cols_to_standardize, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# STEP 11: SAVE DATASET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: SAVING DATASET")
print("=" * 80)

print(f"\nSaving dataset to: {OUTPUT_FILE}")
df_standardized.to_csv(OUTPUT_FILE, index=False)

print("  ✓ Dataset saved successfully!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETE - FINAL DATASET SUMMARY")
print("=" * 80)

print(f"\nDataset: {OUTPUT_FILE}")
print(f"Total observations: {len(df_standardized):,}")
print(f"Total features: {len(df_standardized.columns)}")
print(f"Standardized features: {len(cols_to_standardize)}")

print("\nClass Distribution:")
print(
    f"  Escalation = 1 (conflict): {df_standardized['escalation'].sum():,} ({df_standardized['escalation'].mean() * 100:.2f}%)")
print(
    f"  Escalation = 0 (peace): {(df_standardized['escalation'] == 0).sum():,} ({(1 - df_standardized['escalation'].mean()) * 100:.2f}%)")
print(f"  Imbalance ratio: {(df_standardized['escalation'] == 0).sum() / df_standardized['escalation'].sum():.1f}:1")

print("\nYear Range:")
print(f"  Start: {df_standardized['year'].min():.0f}")
print(f"  End: {df_standardized['year'].max():.0f}")
print(f"  Span: {df_standardized['year'].max() - df_standardized['year'].min():.0f} years")

print("\nData Completeness:")
print(
    f"  Rows with capabilities: {df_standardized[['cinc_a', 'cinc_b']].notna().all(axis=1).sum():,} ({df_standardized[['cinc_a', 'cinc_b']].notna().all(axis=1).mean() * 100:.1f}%)")
print(
    f"  Rows with alliances: {df_standardized['any_alliance'].sum():,} ({df_standardized['any_alliance'].mean() * 100:.1f}%)")
print(
    f"  Rows with trade: {(df_standardized['total_trade'] > 0).sum():,} ({(df_standardized['total_trade'] > 0).mean() * 100:.1f}%)")
print(
    f"  Rows with contiguity: {df_standardized['is_contiguous'].sum():,} ({df_standardized['is_contiguous'].mean() * 100:.1f}%)")

print("\nKey Features (sample):")
sample_features = ['cinc_ratio', 'power_parity', 'milex_ratio', 'any_alliance',
                   'total_trade', 'is_contiguous', 'geographic_distance']
print(df_standardized[sample_features].describe())

print("\n" + "=" * 80)
print("✓ PIPELINE SUCCESSFULLY COMPLETED!")
print("=" * 80)

print("\nNext steps:")
print("  1. Load the dataset: df = pd.read_csv('conflict_escalation_dataset.csv')")
print("  2. Prepare features: X = df[standardized_features]")
print("  3. Prepare target: y = df['escalation']")
print("  4. Train model: LogisticRegression(class_weight='balanced')")
print("\n" + "=" * 80)

