"""Export high-resolution and SVG versions of charts 05 and 06."""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.chdir(r"c:\Users\Admin\Documents\sustainability_project")

print("Regenerating high-res/SVG charts for 05 and 06...")

merged = pd.read_csv(r"CSV_files/Renewables_vs_CO2_2022.csv")

# region map (same fallback)
region_map = {
    "Nigeria": "Africa", "Ghana": "Africa", "Kenya": "Africa", "Ethiopia": "Africa", "Uganda": "Africa",
    "South Africa": "Africa", "Botswana": "Africa", "Zimbabwe": "Africa", "Tanzania": "Africa", "Rwanda": "Africa",
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Poland": "Europe", "Netherlands": "Europe", "Belgium": "Europe", "Austria": "Europe", "Switzerland": "Europe",
    "China": "Asia", "India": "Asia", "Japan": "Asia", "South Korea": "Asia", "Thailand": "Asia",
    "Indonesia": "Asia", "Pakistan": "Asia", "Philippines": "Asia", "Vietnam": "Asia", "Bangladesh": "Asia",
    "Brazil": "South America", "Argentina": "South America", "Chile": "South America", "Colombia": "South America",
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "Australia": "Oceania", "New Zealand": "Oceania"
}
merged['Region'] = merged['Country'].map(region_map)

# compute stats
pearson_r, pearson_p = stats.pearsonr(merged["Renewable %"], merged["CO2 per capita"])
if isinstance(pearson_r, (np.ndarray, list, tuple)):
    pearson_r = float(np.asarray(pearson_r).flatten()[0])
else:
    pearson_r = float(pearson_r)
r_squared = pearson_r ** 2.0
median_renew = merged["Renewable %"].median()
high_renew = merged[merged["Renewable %"] >= median_renew]["CO2 per capita"]
low_renew = merged[merged["Renewable %"] < median_renew]["CO2 per capita"]
t_statistic, ttest_p = stats.ttest_ind(high_renew, low_renew)
co2_no_na = merged["CO2 per capita"].fillna(merged["CO2 per capita"].mean())
renew_no_na = merged["Renewable %"].fillna(merged["Renewable %"].mean())
merged["CO2_zscore"] = np.abs(stats.zscore(co2_no_na.values))
merged["Renew_zscore"] = np.abs(stats.zscore(renew_no_na.values))
outliers = merged[(merged["CO2_zscore"] > 2.5) | (merged["Renew_zscore"] > 2.5)].copy()

# helper to save both png and svg
def savefig_both(fig, base_name, dpi=600):
    png_name = f"{base_name}.png"
    svg_name = f"{base_name}.svg"
    fig.savefig(png_name, dpi=dpi, bbox_inches='tight')
    fig.savefig(svg_name, bbox_inches='tight')

# Create chart 05
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
sns.scatterplot(data=merged, x="Renewable %", y="CO2 per capita",
                hue="Region", style="Region", s=100, edgecolor="black", alpha=0.6, ax=ax)
if len(outliers) > 0:
    ax.scatter(outliers["Renewable %"], outliers["CO2 per capita"],
               color="orange", marker="D", s=200, edgecolor="darkorange", linewidth=2.5, label="Outliers", zorder=5)
    for _, row in outliers.head(8).iterrows():
        ax.annotate(row["Country"], (row["Renewable %"], row["CO2 per capita"]), xytext=(5,5), textcoords='offset points', fontsize=9, color='darkorange')
ax.set_xlabel("Renewable energy consumption (%)")
ax.set_ylabel("CO2 emissions per capita (metric tons)")
ax.set_title("Clutch Moments: Outliers & Extreme Cases")
ax.grid(True, alpha=0.3)

ax = axes[1]
colors = ['orange' if (abs(zco2) > 2.5 or abs(zren) > 2.5) else 'blue' for zco2, zren in zip(merged["CO2_zscore"], merged["Renew_zscore"]) ]
ax.scatter(merged["CO2_zscore"], merged["Renew_zscore"], c=colors, s=100, alpha=0.6, edgecolor='black')
ax.axhline(2.5, color='red', linestyle='--')
ax.axvline(2.5, color='red', linestyle='--')
ax.set_xlabel("CO2 Z-score")
ax.set_ylabel("Renewable % Z-score")
ax.set_title("Z-score Distribution (Outlier Detection)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig_both(fig, "05_Outliers_and_Clutch_Moments")
plt.close(fig)
print('Saved 05 high-res PNG and SVG')

# Create chart 06
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0,0])
ax1.text(0.5, 0.9, 'H1: CORRELATION TEST', ha='center')
ax1.text(0.5, 0.7, f'Pearson r: {pearson_r:.4f}', ha='center')
ax1.text(0.5, 0.55, f'p-value: {pearson_p:.6f}', ha='center')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0,1])
ax2.text(0.5, 0.95, 'VARIANCE EXPLAINED', ha='center')
ax2.pie([r_squared*100, (1-r_squared)*100], labels=[f'Explained\n{r_squared*100:.1f}%', f'Unexplained\n{(1-r_squared)*100:.1f}%'], colors=['lightgreen','lightcoral'], startangle=90)
ax2.axis('off')

ax3 = fig.add_subplot(gs[0,2])
ax3.text(0.5, 0.9, 'H2: REGIONAL ANOVA', ha='center')
ax3.axis('off')

ax4 = fig.add_subplot(gs[1,0])
ax4.text(0.5, 0.9, 'H3: t-TEST', ha='center')
ax4.axis('off')

ax5 = fig.add_subplot(gs[1,1])
ax5.text(0.5, 0.9, f'Total Countries: {len(merged)}', ha='center')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1,2])
ax6.text(0.5, 0.9, f'Pearson r: {abs(pearson_r):.4f}', ha='center')
ax6.axis('off')

ax7 = fig.add_subplot(gs[2,:])
findings_text = "Key findings reproduced from analysis."
ax7.text(0.05, 0.5, findings_text)
ax7.axis('off')

plt.tight_layout()
savefig_both(fig, "06_Summary_Statistics_Dashboard")
plt.close(fig)
print('Saved 06 high-res PNG and SVG')

print('High-resolution export complete.')
