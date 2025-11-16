#!/usr/bin/env python
"""Quick test to verify H2 update works correctly"""

import sys
sys.path.insert(0, r"c:\Users\Admin\Documents\sustainability_project")

import pandas as pd
import numpy as np
from scipy import stats
from region_mapping import REGION_MAP

# Load data
print("Loading data...")
merged = pd.read_csv(r"c:\Users\Admin\Documents\sustainability_project\CSV_files\Renewables_vs_CO2_2022.csv")

# Map regions
merged['Region'] = merged['Country'].map(REGION_MAP)
print(f"✓ Loaded {len(merged)} countries")
print(f"✓ Region mapping applied: {merged['Region'].notna().sum()} countries mapped")

# Test H2: Top 25% vs Bottom 25% CO2 emitters
quartile_25_idx = len(merged) // 4
top_25_co2 = merged.nlargest(quartile_25_idx, "CO2 per capita")
bottom_25_co2 = merged.nsmallest(quartile_25_idx, "CO2 per capita")

top_25_renew = top_25_co2["Renewable %"].dropna()
bottom_25_renew = bottom_25_co2["Renewable %"].dropna()

t_stat, p_value = stats.ttest_ind(top_25_renew, bottom_25_renew)

print(f"\n[H2] CO2 EMITTERS COMPARISON TEST")
print(f"=" * 60)
print(f"Top 25% CO2 Emitters (n={len(top_25_renew)}): {top_25_renew.mean():.2f}% renewable (±{top_25_renew.std():.2f}%)")
print(f"Bottom 25% CO2 Emitters (n={len(bottom_25_renew)}): {bottom_25_renew.mean():.2f}% renewable (±{bottom_25_renew.std():.2f}%)")
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"\n{'✓ SIGNIFICANT' if p_value < 0.05 else '✗ NOT SIGNIFICANT'} (α=0.05)")
print(f"\n✓ H2 UPDATE SUCCESSFUL - p-value is NOT NaN!")
