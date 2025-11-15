# ============================================================================
# SUSTAINABILITY ANALYSIS: CO2 Emissions vs Renewable Energy (2022)
# World Bank Data Analysis with Hypothesis Testing
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# load datasets
co2 = pd.read_csv("API_EN.GHG.CO2.PC.CE.AR5_DS2_en_csv_v2_127841.csv", skiprows=4)
renew = pd.read_csv("API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_130800.csv", skiprows=4)

# Constants
YEAR = "2022"
SIGNIFICANCE_LEVEL = 0.05  # 5% significance level

# Rename for consistency
co2.rename(columns={"Country Name": "Country"}, inplace=True)
renew.rename(columns={"Country Name": "Country"}, inplace=True)

# Select relevant columns
co2 = co2[["Country", YEAR]].rename(columns={YEAR: "CO2 per capita"})
renew = renew[["Country", YEAR]].rename(columns={YEAR: "Renewable %"})

# Merge datasets
merged = pd.merge(co2, renew, on="Country", how="inner")

# Convert to numeric
merged["CO2 per capita"] = pd.to_numeric(merged["CO2 per capita"], errors="coerce")
merged["Renewable %"] = pd.to_numeric(merged["Renewable %"], errors="coerce")

# Drop rows with missing values
merged.dropna(subset=["CO2 per capita", "Renewable %"], inplace=True)

# Add regional grouping (expand as needed)
region_map = {
    "Nigeria": "Africa", "Ghana": "Africa", "Kenya": "Africa",
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe",
    "China": "Asia", "India": "Asia", "Japan": "Asia",
    "Brazil": "South America", "Argentina": "South America",
    "United States": "North America", "Canada": "North America",
    "Australia": "Oceania"
}

merged["Region"] = merged["Country"].map(region_map)

regional_avg = (
    merged.dropna(subset=["Region"])
          .groupby("Region")[["CO2 per capita", "Renewable %"]]
          .mean()
          .reset_index()
)

# ============================================================================
# RESEARCH HYPOTHESES
# ============================================================================
print("="*80)
print("RESEARCH HYPOTHESES FOR SUSTAINABILITY ANALYSIS")
print("="*80)

hypotheses = {
    "H1": {
        "name": "Primary Hypothesis: Negative Correlation",
        "description": "H1: Countries with HIGHER renewable energy consumption have LOWER CO2 emissions per capita",
        "null": "H0: There is NO significant correlation between renewable energy and CO2 emissions",
        "type": "Two-tailed Pearson correlation test"
    },
    "H2": {
        "name": "Secondary Hypothesis: Regional Disparities",
        "description": "H2: There are SIGNIFICANT differences in CO2 emissions across regions despite varying renewable adoption",
        "null": "H0: All regions have equal mean CO2 emissions",
        "type": "One-way ANOVA test"
    },
    "H3": {
        "name": "Tertiary Hypothesis: Renewable Effectiveness",
        "description": "H3: High renewable countries have statistically different CO2 levels than low renewable countries",
        "null": "H0: High and low renewable adoption groups have equal CO2 emissions",
        "type": "Independent t-test"
    }
}

for h_id, h_details in hypotheses.items():
    print(f"\n{h_id}: {h_details['name']}")
    print(f"  Description: {h_details['description']}")
    print(f"  Null Hypothesis: {h_details['null']}")
    print(f"  Test Type: {h_details['type']}")

print("\n" + "="*80)

# ============================================================================
# HYPOTHESIS TESTING & STATISTICAL ANALYSIS
# ============================================================================

print("\nSTATISTICAL TEST RESULTS")
print("="*80)

# === H1: Test Correlation ===
print("\n[H1] CORRELATION TEST: Renewable Energy vs CO2 Emissions")
print("-"*80)

pearson_r, pearson_p = pearsonr(merged["Renewable %"], merged["CO2 per capita"])
spearman_r, spearman_p = spearmanr(merged["Renewable %"], merged["CO2 per capita"])

print(f"Pearson Correlation Coefficient: r = {pearson_r:.4f}")
print(f"P-value: {pearson_p:.6f}")
print(f"Significance Level: alpha = {SIGNIFICANCE_LEVEL}")

if pearson_p < SIGNIFICANCE_LEVEL:
    print(f"[+] RESULT: REJECT NULL HYPOTHESIS (p < {SIGNIFICANCE_LEVEL})")
    print(f"  -> There IS a statistically significant correlation between renewable energy and CO2 emissions")
else:
    print(f"[-] RESULT: FAIL TO REJECT NULL HYPOTHESIS (p >= {SIGNIFICANCE_LEVEL})")
    print(f"  -> No statistically significant correlation found")

print(f"\nSpearman Correlation Coefficient: rho = {spearman_r:.4f}")
print(f"P-value: {spearman_p:.6f}")

if pearson_r < 0:
    print(f"\n[+] INTERPRETATION: Strong NEGATIVE correlation detected")
    print(f"  -> As renewable % increases, CO2 emissions tend to DECREASE")
else:
    print(f"\n[+] INTERPRETATION: Positive correlation detected")

# Calculate R-squared (coefficient of determination)
r_squared = pearson_r ** 2
print(f"\nCoefficient of Determination (R^2): {r_squared:.4f}")
print(f"  -> {r_squared*100:.2f}% of CO2 variance is explained by renewable energy adoption")

# === H2: Regional Disparities (ANOVA) ===
print("\n[H2] REGIONAL DISPARITIES TEST: One-way ANOVA")
print("-"*80)

# Filter to only include regions that exist in data
regions_with_data = merged[merged["Region"].notna()].groupby("Region")["CO2 per capita"].apply(list).to_dict()

if len(regions_with_data) >= 2:
    regional_groups = list(regions_with_data.values())
    f_statistic, anova_p = f_oneway(*regional_groups)
    
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"P-value: {anova_p:.6f}")
    print(f"Significance Level: alpha = {SIGNIFICANCE_LEVEL}")
    
    if anova_p < SIGNIFICANCE_LEVEL:
        print(f"[+] RESULT: REJECT NULL HYPOTHESIS (p < {SIGNIFICANCE_LEVEL})")
        print(f"  -> There ARE statistically significant differences in CO2 emissions across regions")
    else:
        print(f"[-] RESULT: FAIL TO REJECT NULL HYPOTHESIS (p >= {SIGNIFICANCE_LEVEL})")
        print(f"  -> No significant regional differences detected")
else:
    print(f"[!] WARNING: Only {len(regions_with_data)} regions with data. ANOVA requires at least 2 groups.")
    print(f"Regions identified in data: {list(regions_with_data.keys())}")
    f_statistic, anova_p = np.nan, np.nan

# === H3: High vs Low Renewable Adoption ===
print("\n[H3] RENEWABLE EFFECTIVENESS TEST: Independent t-test")
print("-"*80)

# Split into high and low renewable adoption
median_renew = merged["Renewable %"].median()
high_renew = merged[merged["Renewable %"] >= median_renew]["CO2 per capita"]
low_renew = merged[merged["Renewable %"] < median_renew]["CO2 per capita"]

t_statistic, ttest_p = ttest_ind(high_renew, low_renew)

print(f"High Renewable Adoption (>={median_renew:.2f}%): n={len(high_renew)}, mean CO2={high_renew.mean():.4f}")
print(f"Low Renewable Adoption (<{median_renew:.2f}%): n={len(low_renew)}, mean CO2={low_renew.mean():.4f}")
print(f"Mean Difference: {abs(high_renew.mean() - low_renew.mean()):.4f} metric tons per capita")
print(f"\nT-statistic: {t_statistic:.4f}")
print(f"P-value: {ttest_p:.6f}")
print(f"Significance Level: alpha = {SIGNIFICANCE_LEVEL}")

if ttest_p < SIGNIFICANCE_LEVEL:
    print(f"[+] RESULT: REJECT NULL HYPOTHESIS (p < {SIGNIFICANCE_LEVEL})")
    print(f"  -> High renewable countries have SIGNIFICANTLY different CO2 emissions than low renewable countries")
else:
    print(f"[-] RESULT: FAIL TO REJECT NULL HYPOTHESIS (p >= {SIGNIFICANCE_LEVEL})")

print("\n" + "="*80)

# ============================================================================
# DISPARITY & OUTLIER ANALYSIS (Clutch Moments)
# ============================================================================
print("\nDISPARITY & OUTLIER ANALYSIS")
print("="*80)

# Calculate z-scores to identify outliers (only for non-NaN values)
z_scores_co2 = np.abs(stats.zscore(merged["CO2 per capita"].dropna(), nan_policy='omit'))
z_scores_renew = np.abs(stats.zscore(merged["Renewable %"].dropna(), nan_policy='omit'))

# Create temporary series with same indices
merged["CO2_zscore"] = np.nan
merged["Renew_zscore"] = np.nan

merged.loc[merged["CO2 per capita"].notna(), "CO2_zscore"] = z_scores_co2
merged.loc[merged["Renewable %"].notna(), "Renew_zscore"] = z_scores_renew

# Outliers (z-score > 2.5)
outliers = merged[(merged["CO2_zscore"] > 2.5) | (merged["Renew_zscore"] > 2.5)].copy()

print(f"\nOUTLIERS DETECTED (Clutch Moments - countries with extreme values):")
print(f"Total outliers: {len(outliers)}")

if len(outliers) > 0:
    print("\nTop Outliers:")
    outlier_display = outliers[["Country", "CO2 per capita", "Renewable %"]].drop_duplicates().sort_values("Country")
    print(outlier_display.head(10))
else:
    print("No extreme outliers detected using z-score threshold of 2.5")

# Global disparity
global_mean_co2 = merged["CO2 per capita"].mean()
global_mean_renew = merged["Renewable %"].mean()
if global_mean_co2 != 0:
    global_cv_co2 = (merged["CO2 per capita"].std() / global_mean_co2) * 100
else:
    global_cv_co2 = 0
if global_mean_renew != 0:
    global_cv_renew = (merged["Renewable %"].std() / global_mean_renew) * 100
else:
    global_cv_renew = 0

print("\nDISPARITY INDEX (Coefficient of Variation):")
print("-"*80)
print(f"{'GLOBAL':15} | CO2 CV: {global_cv_co2:6.2f}% | Renewable CV: {global_cv_renew:6.2f}%")

print("\n" + "="*80)

# ============================================================================
# VISUALIZATION 1: Main Scatter Plot with Statistical Annotations
# ============================================================================
print("\nGenerating VISUALIZATION 1: Main Correlation Plot with Statistical Results...")

plt.figure(figsize=(14, 8))

# Scatter plot by region
sns.scatterplot(data=merged, x="Renewable %", y="CO2 per capita",
                hue="Region", style="Region", s=100, edgecolor="black", alpha=0.7)

# Regression line with 95% confidence interval
sns.regplot(data=merged, x="Renewable %", y="CO2 per capita",
            scatter=False, color="blue", line_kws={"linewidth": 2.5, "linestyle": "--"},
            ci=95)

# Regional averages
if len(regional_avg) > 0:
    plt.scatter(regional_avg["Renewable %"], regional_avg["CO2 per capita"],
                color="red", marker="*", s=500, label="Regional Average", edgecolor="darkred", linewidth=2)

# Highlight outliers
if len(outliers) > 0:
    plt.scatter(outliers["Renewable %"], outliers["CO2 per capita"],
                color="orange", marker="D", s=150, label="Outliers (Clutch Moments)", 
                edgecolor="darkorange", linewidth=2, alpha=0.8)

# Add statistical annotation
textstr = f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})\n'
textstr += f'R^2 = {r_squared:.3f}\n'
if pearson_p < SIGNIFICANCE_LEVEL:
    textstr += f'*** SIGNIFICANT (p < {SIGNIFICANCE_LEVEL}) ***'
else:
    textstr += f'Not significant (p >= {SIGNIFICANCE_LEVEL})'

plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontweight='bold')

# Labels and title
plt.xlabel("Renewable energy consumption (%)", fontsize=13, fontweight='bold')
plt.ylabel("CO2 emissions per capita (metric tons)", fontsize=13, fontweight='bold')
plt.title("H1: Renewable Energy vs CO2 Emissions\n(With Statistical Significance)", 
          fontsize=15, fontweight='bold')
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("01_Main_Correlation_Analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Chart saved as: 01_Main_Correlation_Analysis.png")

# ============================================================================
# VISUALIZATION 2: Data Distribution Analysis
# ============================================================================
print("\nGenerating VISUALIZATION 2: Data Distribution Analysis (H2)...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 2a: Histogram of CO2
axes[0, 0].hist(merged["CO2 per capita"], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title("Distribution of CO2 Emissions\n(All Countries)", fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Frequency", fontsize=11)
axes[0, 0].set_xlabel("CO2 per capita (metric tons)", fontsize=11)
axes[0, 0].axvline(merged["CO2 per capita"].mean(), color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {merged["CO2 per capita"].mean():.2f}')
axes[0, 0].legend()

# 2b: Histogram of Renewable %
axes[0, 1].hist(merged["Renewable %"], bins=20, color='seagreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_title("Distribution of Renewable Energy Adoption\n(All Countries)", fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel("Frequency", fontsize=11)
axes[0, 1].set_xlabel("Renewable % of total energy", fontsize=11)
axes[0, 1].axvline(merged["Renewable %"].mean(), color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {merged["Renewable %"].mean():.2f}')
axes[0, 1].legend()

# 2c: Scatter of CO2 by Renewable % (quartile grouping)
merged_clean = merged.dropna(subset=["CO2 per capita", "Renewable %"])
merged_clean["Renewable_Quartile"] = pd.qcut(merged_clean["Renewable %"], q=4, 
                                             labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"], 
                                             duplicates='drop')

# Box plot by renewable quartile
quartile_order = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
existing_quartiles = [q for q in quartile_order if q in merged_clean["Renewable_Quartile"].values]

if len(existing_quartiles) > 1:
    bp_data = [merged_clean[merged_clean["Renewable_Quartile"] == q]["CO2 per capita"].values for q in existing_quartiles]
    bp = axes[1, 0].boxplot(bp_data, labels=existing_quartiles, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
else:
    axes[1, 0].hist(merged["CO2 per capita"], bins=15, color='lightblue', edgecolor='black', alpha=0.7)
    
axes[1, 0].set_title("CO2 Emissions by Renewable Energy Quartile\n(H2 Disparity Analysis)", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("CO2 per capita (metric tons)", fontsize=11)
axes[1, 0].set_xlabel("Renewable Energy Adoption Level", fontsize=11)

# 2d: Cumulative distribution
sorted_co2 = np.sort(merged["CO2 per capita"])
cumulative = np.arange(1, len(sorted_co2) + 1) / len(sorted_co2)
axes[1, 1].plot(sorted_co2, cumulative * 100, linewidth=2.5, color='steelblue')
axes[1, 1].set_title("Cumulative Distribution of CO2 Emissions", fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel("Cumulative %", fontsize=11)
axes[1, 1].set_xlabel("CO2 per capita (metric tons)", fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='Median (50%)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("02_Regional_Disparities_Analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Chart saved as: 02_Regional_Disparities_Analysis.png")

# ============================================================================
# VISUALIZATION 3: High vs Low Renewable Adoption (H3)
# ============================================================================
print("\nGenerating VISUALIZATION 3: Renewable Effectiveness Analysis (H3)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 3a: Grouped bar chart
groups_data = pd.DataFrame({
    'Category': ['High Renewable\n(>= median)', 'Low Renewable\n(< median)'],
    'Mean CO2': [high_renew.mean(), low_renew.mean()],
    'Std Dev': [high_renew.std(), low_renew.std()]
})

colors = ['green', 'red']
axes[0].bar(groups_data['Category'], groups_data['Mean CO2'], 
            yerr=groups_data['Std Dev'], capsize=15, color=colors, alpha=0.7, 
            edgecolor='black', linewidth=2)
axes[0].set_ylabel('Mean CO2 per capita (metric tons)', fontsize=12, fontweight='bold')
axes[0].set_title('H3: CO2 Emissions by Renewable Adoption Level\n(Independent t-test)', 
                  fontsize=12, fontweight='bold')

# Add t-test result annotation
ttest_text = f"n_high = {len(high_renew)}, n_low = {len(low_renew)}\n"
ttest_text += f"t-statistic = {t_statistic:.3f}\n"
ttest_text += f"p-value = {ttest_p:.4f}\n"
if ttest_p < SIGNIFICANCE_LEVEL:
    ttest_text += f"*** SIGNIFICANT (p < {SIGNIFICANCE_LEVEL}) ***"
else:
    ttest_text += f"Not significant"
axes[0].text(0.98, 0.97, ttest_text, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), fontweight='bold')

# 3b: Distribution comparison
axes[1].hist(high_renew, bins=15, alpha=0.6, label='High Renewable', color='green', edgecolor='black')
axes[1].hist(low_renew, bins=15, alpha=0.6, label='Low Renewable', color='red', edgecolor='black')
axes[1].axvline(high_renew.mean(), color='darkgreen', linestyle='--', linewidth=2.5, 
               label=f'High Mean: {high_renew.mean():.2f}')
axes[1].axvline(low_renew.mean(), color='darkred', linestyle='--', linewidth=2.5, 
               label=f'Low Mean: {low_renew.mean():.2f}')
axes[1].set_xlabel('CO2 per capita (metric tons)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution of CO2 Emissions:\nHigh vs Low Renewable Adoption', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("03_High_vs_Low_Renewable_Analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Chart saved as: 03_High_vs_Low_Renewable_Analysis.png")

# ============================================================================
# VISUALIZATION 4: Top 10 Countries
# ============================================================================
print("\nGenerating VISUALIZATION 4: Top 10 Countries Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 10 CO2 emitters
top_emitters = merged.nlargest(10, "CO2 per capita")
sns.barplot(data=top_emitters, x="CO2 per capita", y="Country", ax=axes[0], 
            palette="Reds_r", edgecolor="black")
axes[0].set_title("Top 10 CO2 Emitters per Capita (2022)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("CO2 emissions (metric tons per capita)", fontsize=11)
axes[0].tick_params(axis='y', labelsize=10)

# Top 10 renewable leaders
top_renewables = merged.nlargest(10, "Renewable %")
sns.barplot(data=top_renewables, x="Renewable %", y="Country", ax=axes[1], 
            palette="Greens_r", edgecolor="black")
axes[1].set_title("Top 10 Renewable Energy Leaders (2022)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Renewable energy (% of total)", fontsize=11)
axes[1].tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig("04_Top_10_Countries_Comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Chart saved as: 04_Top_10_Countries_Comparison.png")

# ============================================================================
# VISUALIZATION 5: Outliers & Clutch Moments
# ============================================================================
print("\nGenerating VISUALIZATION 5: Outliers & Clutch Moments...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 5a: Scatter with outliers
ax = axes[0]
sns.scatterplot(data=merged, x="Renewable %", y="CO2 per capita",
                hue="Region", style="Region", s=100, edgecolor="black", alpha=0.6, ax=ax)

if len(outliers) > 0:
    ax.scatter(outliers["Renewable %"], outliers["CO2 per capita"],
               color="orange", marker="D", s=200, edgecolor="darkorange", linewidth=2.5, 
               label="Outliers", zorder=5)
    
    for _, row in outliers.head(8).iterrows():
        ax.annotate(row["Country"], 
                   (row["Renewable %"], row["CO2 per capita"]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='darkorange', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

ax.set_xlabel("Renewable energy consumption (%)", fontsize=12, fontweight='bold')
ax.set_ylabel("CO2 emissions per capita (metric tons)", fontsize=12, fontweight='bold')
ax.set_title("Clutch Moments: Outliers & Extreme Cases", fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# 5b: Z-score distribution
ax = axes[1]
merged_zscores = merged.dropna(subset=["CO2_zscore", "Renew_zscore"])
colors = ['orange' if (abs(zco2) > 2.5 or abs(zren) > 2.5) else 'blue' 
          for zco2, zren in zip(merged_zscores["CO2_zscore"], merged_zscores["Renew_zscore"])]
ax.scatter(merged_zscores["CO2_zscore"], merged_zscores["Renew_zscore"], c=colors, s=100, alpha=0.6, edgecolor='black')
ax.axhline(2.5, color='red', linestyle='--', linewidth=2, label='Outlier Threshold (z=2.5)')
ax.axhline(-2.5, color='red', linestyle='--', linewidth=2)
ax.axvline(2.5, color='red', linestyle='--', linewidth=2)
ax.axvline(-2.5, color='red', linestyle='--', linewidth=2)
ax.set_xlabel("CO2 Z-score", fontsize=12, fontweight='bold')
ax.set_ylabel("Renewable % Z-score", fontsize=12, fontweight='bold')
ax.set_title("Z-score Distribution (Outlier Detection)", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("05_Outliers_and_Clutch_Moments.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Chart saved as: 05_Outliers_and_Clutch_Moments.png")

# ============================================================================
# VISUALIZATION 6: Summary Statistics Dashboard
# ============================================================================
print("\nGenerating VISUALIZATION 6: Summary Statistics Dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('HYPOTHESIS TESTING RESULTS SUMMARY', fontsize=16, fontweight='bold', y=0.98)

# 6a: H1 Correlation
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.9, 'H1: CORRELATION TEST', ha='center', fontsize=12, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.5, 0.7, f'Pearson r: {pearson_r:.4f}', ha='center', fontsize=11, transform=ax1.transAxes)
ax1.text(0.5, 0.55, f'p-value: {pearson_p:.6f}', ha='center', fontsize=11, transform=ax1.transAxes)
result_h1 = "[+] REJECT H0" if pearson_p < SIGNIFICANCE_LEVEL else "[-] FAIL TO REJECT H0"
color_h1 = 'lightgreen' if pearson_p < SIGNIFICANCE_LEVEL else 'lightcoral'
ax1.text(0.5, 0.35, result_h1, ha='center', fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor=color_h1, alpha=0.8), transform=ax1.transAxes)
ax1.axis('off')

# 6b: R-squared
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.85, 'VARIANCE EXPLAINED', ha='center', fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.pie([r_squared*100, (1-r_squared)*100], labels=[f'Explained\n{r_squared*100:.1f}%', f'Unexplained\n{(1-r_squared)*100:.1f}%'],
        colors=['lightgreen', 'lightcoral'], startangle=90, ax=ax2, textprops={'fontsize': 10, 'fontweight': 'bold'})

# 6c: H2 ANOVA
ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.9, 'H2: REGIONAL ANOVA', ha='center', fontsize=12, fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.7, f'F-stat: {f_statistic:.4f}', ha='center', fontsize=11, transform=ax3.transAxes)
ax3.text(0.5, 0.55, f'p-value: {anova_p:.6f}', ha='center', fontsize=11, transform=ax3.transAxes)
result_h2 = "[+] REJECT H0" if (not np.isnan(anova_p) and anova_p < SIGNIFICANCE_LEVEL) else "[-] FAIL TO REJECT H0"
color_h2 = 'lightgreen' if (not np.isnan(anova_p) and anova_p < SIGNIFICANCE_LEVEL) else 'lightcoral'
ax3.text(0.5, 0.35, result_h2, ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=color_h2, alpha=0.8), transform=ax3.transAxes)
ax3.axis('off')

# 6d: H3 t-test
ax4 = fig.add_subplot(gs[1, 0])
ax4.text(0.5, 0.9, 'H3: t-TEST', ha='center', fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.5, 0.7, f't-stat: {t_statistic:.4f}', ha='center', fontsize=11, transform=ax4.transAxes)
ax4.text(0.5, 0.55, f'p-value: {ttest_p:.6f}', ha='center', fontsize=11, transform=ax4.transAxes)
result_h3 = "[+] REJECT H0" if ttest_p < SIGNIFICANCE_LEVEL else "[-] FAIL TO REJECT H0"
color_h3 = 'lightgreen' if ttest_p < SIGNIFICANCE_LEVEL else 'lightcoral'
ax4.text(0.5, 0.35, result_h3, ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=color_h3, alpha=0.8), transform=ax4.transAxes)
ax4.axis('off')

# 6e: Sample Info
ax5 = fig.add_subplot(gs[1, 1])
ax5.text(0.5, 0.9, 'SAMPLE INFORMATION', ha='center', fontsize=12, fontweight='bold', transform=ax5.transAxes)
ax5.text(0.5, 0.75, f'Total Countries: {len(merged)}', ha='center', fontsize=11, transform=ax5.transAxes)
ax5.text(0.5, 0.60, f'Regions Mapped: {merged["Region"].nunique()}', ha='center', fontsize=11, transform=ax5.transAxes)
ax5.text(0.5, 0.45, f'Outliers: {len(outliers)}', ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), transform=ax5.transAxes)
ax5.axis('off')

# 6f: Effect Sizes
ax6 = fig.add_subplot(gs[1, 2])
ax6.text(0.5, 0.9, 'EFFECT SIZES', ha='center', fontsize=12, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.75, f'Pearson r: {abs(pearson_r):.4f}', ha='center', fontsize=10, transform=ax6.transAxes)
ax6.text(0.5, 0.60, f'R^2: {r_squared:.4f}', ha='center', fontsize=10, transform=ax6.transAxes)
cohens_d = abs(high_renew.mean() - low_renew.mean()) / np.sqrt((high_renew.std()**2 + low_renew.std()**2) / 2)
ax6.text(0.5, 0.45, f"Cohen's d: {cohens_d:.4f}", ha='center', fontsize=10, transform=ax6.transAxes)
ax6.axis('off')

# Bottom: Key Findings
ax7 = fig.add_subplot(gs[2, :])
findings_text = f"""KEY FINDINGS & INTERPRETATION:
- H1 (Correlation): {'Strong negative correlation detected' if pearson_r < -0.5 else 'Moderate/weak correlation observed'}
- H2 (Regional Disparities): {'Significant differences exist' if (not np.isnan(anova_p) and anova_p < SIGNIFICANCE_LEVEL) else 'No significant regional differences'}
- H3 (Renewable Effectiveness): {'High renewable adoption countries have significantly lower CO2' if ttest_p < SIGNIFICANCE_LEVEL else 'No significant difference'}
- Overall: {f'{r_squared*100:.1f}% of CO2 variation explained by renewable energy' if r_squared > 0.3 else 'Renewable energy is one of several factors'}"""
ax7.text(0.05, 0.5, findings_text, fontsize=11, transform=ax7.transAxes, 
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, pad=1))
ax7.axis('off')

plt.savefig("06_Summary_Statistics_Dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Chart saved as: 06_Summary_Statistics_Dashboard.png")

# ============================================================================
# SAVE OUTPUTS & GENERATE REPORT
# ============================================================================
print("\n" + "="*80)
print("SAVING DATA & GENERATING REPORT")
print("="*80)

merged.to_csv("Renewables_vs_CO2_2022.csv", index=False)
regional_avg.to_csv("Regional_Averages_2022.csv", index=False)
outliers.to_csv("Outliers_and_Extreme_Cases.csv", index=False)

# Generate detailed report
report = f"""{'='*80}
SUSTAINABILITY ANALYSIS REPORT: CO2 vs RENEWABLE ENERGY (2022)
{'='*80}

1. RESEARCH HYPOTHESES
{'-'*80}
H1: Countries with HIGHER renewable energy consumption have LOWER CO2 emissions
H2: There are SIGNIFICANT differences in CO2 emissions across regions
H3: High renewable adoption countries have LOWER CO2 than low renewable countries

2. STATISTICAL TEST RESULTS
{'-'*80}

H1 - CORRELATION ANALYSIS:
  Pearson Correlation: r = {pearson_r:.6f}
  P-value: {pearson_p:.8f}
  Significance Level: alpha = {SIGNIFICANCE_LEVEL}
  Result: {'[+] SIGNIFICANT' if pearson_p < SIGNIFICANCE_LEVEL else '[-] NOT SIGNIFICANT'}
  Interpretation: {f'Strong negative correlation' if pearson_r < -0.5 else 'Moderate correlation' if pearson_r < -0.3 else 'Weak correlation'}
  R^2 = {r_squared:.4f} ({r_squared*100:.2f}% variance explained)

H2 - ANOVA (REGIONAL DISPARITIES):
  F-statistic: {f_statistic:.6f}
  P-value: {anova_p:.8f}
  Significance Level: alpha = {SIGNIFICANCE_LEVEL}
  Result: {'[+] SIGNIFICANT' if (not np.isnan(anova_p) and anova_p < SIGNIFICANCE_LEVEL) else '[-] NOT SIGNIFICANT'}
  
H3 - INDEPENDENT t-TEST:
  t-statistic: {t_statistic:.6f}
  P-value: {ttest_p:.8f}
  Significance Level: alpha = {SIGNIFICANCE_LEVEL}
  Result: {'[+] SIGNIFICANT' if ttest_p < SIGNIFICANCE_LEVEL else '[-] NOT SIGNIFICANT'}
  High Renewable Mean CO2: {high_renew.mean():.4f}
  Low Renewable Mean CO2: {low_renew.mean():.4f}
  Difference: {abs(high_renew.mean() - low_renew.mean()):.4f} metric tons

3. DISPARITY ANALYSIS
{'-'*80}
Total Countries Analyzed: {len(merged)}
Regions Identified: {merged['Region'].nunique()}
Outliers Detected: {len(outliers)}
Global CO2 Disparity Index (CV): {global_cv_co2:.2f}%
Global Renewable Disparity Index (CV): {global_cv_renew:.2f}%

4. OUTLIERS & CLUTCH MOMENTS
{'-'*80}
{outliers[['Country', 'CO2 per capita', 'Renewable %']].to_string() if len(outliers) > 0 else 'No extreme outliers detected'}

5. CONCLUSIONS
{'-'*80}
- The analysis {'confirms' if pearson_p < SIGNIFICANCE_LEVEL and pearson_r < 0 else 'does not confirm'} that renewable energy adoption is associated with lower CO2
- Regional disparities in CO2 emissions {'are' if (not np.isnan(anova_p) and anova_p < SIGNIFICANCE_LEVEL) else 'are not'} statistically significant
- Countries with higher renewable adoption {'do' if ttest_p < SIGNIFICANCE_LEVEL else 'do not'} have significantly lower CO2 per capita
- Renewable energy explains {r_squared*100:.2f}% of the variation in CO2 emissions

{'='*80}
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("Analysis_Report.txt", "w") as f:
    f.write(report)

print("\n[OK] Data files saved:")
print("     - Renewables_vs_CO2_2022.csv")
print("     - Regional_Averages_2022.csv")
print("     - Outliers_and_Extreme_Cases.csv")
print("\n[OK] Report saved: Analysis_Report.txt")
print("\n[OK] Visualizations saved:")
print("     - 01_Main_Correlation_Analysis.png")
print("     - 02_Regional_Disparities_Analysis.png")
print("     - 03_High_vs_Low_Renewable_Analysis.png")
print("     - 04_Top_10_Countries_Comparison.png")
print("     - 05_Outliers_and_Clutch_Moments.png")
print("     - 06_Summary_Statistics_Dashboard.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
