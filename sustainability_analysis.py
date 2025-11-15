# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load datasets
co2 = pd.read_csv("API_EN.GHG.CO2.PC.CE.AR5_DS2_en_csv_v2_127841.csv", skiprows=4)
renew = pd.read_csv("API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_130800.csv", skiprows=4)
co2.head()
renew.head()
# Constants
YEAR = "2022"

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
# Print regional averages
print("\n--- Regional Averages ---")
print("\n--- Correlation Matrix ---")
print(merged[["CO2 per capita", "Renewable %"]].corr())

sns.set_theme(style="whitegrid", palette="muted")
plt.figure(figsize=(12, 7))

# Scatter plot by region
sns.scatterplot(data=merged, x="Renewable %", y="CO2 per capita",
                hue="Region", style="Region", s=80, edgecolor="black", alpha=0.75)

# Regression line
sns.regplot(data=merged, x="Renewable %", y="CO2 per capita",
            scatter=False, color="blue", line_kws={"linewidth": 2, "linestyle": "--"})

# Regional averages
plt.scatter(regional_avg["Renewable %"], regional_avg["CO2 per capita"],
            color="red", marker="*", s=300, label="Regional Average")

# Annotate regions
for _, row in regional_avg.iterrows():
    plt.text(row["Renewable %"] + 0.5, row["CO2 per capita"] + 0.2,
             row["Region"], fontsize=10, color="red")

# Labels and title
plt.xlabel("Renewable energy consumption (%)", fontsize=12)
plt.ylabel("CO2 emissions per capita (metric tons)", fontsize=12)
plt.title("Renewables vs CO2 Emissions (World Bank, 2022)", fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig("Renewables_vs_CO2_2022.png", dpi=300)
# --- Automatically label top 5 emitters and top 5 renewable leaders with arrows ---

# Top 5 CO2 emitters per capita
top_emitters = merged.nlargest(5, "CO2 per capita")

# Top 5 renewable leaders
top_renewables = merged.nlargest(5, "Renewable %")

# Annotate emitters in red with arrows
for _, row in top_emitters.iterrows():
    plt.annotate(row["Country"],
                 (row["Renewable %"], row["CO2 per capita"]),
                 xytext=(row["Renewable %"]+5, row["CO2 per capita"]+2),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=9, color="red")

# Annotate renewable leaders in green with arrows
for _, row in top_renewables.iterrows():
    plt.annotate(row["Country"],
                 (row["Renewable %"], row["CO2 per capita"]),
                 xytext=(row["Renewable %"]+5, row["CO2 per capita"]+2),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=9, color="green")

plt.show()

merged.to_csv("Renewables_vs_CO2_2022.csv", index=False)
regional_avg.to_csv("Regional_Averages_2022.csv", index=False)
print(regional_avg)

# Top 10 emitters
top_emitters = merged.nlargest(10, "CO2 per capita")

# Top 10 renewable leaders
top_renewables = merged.nlargest(10, "Renewable %")

# Plot side-by-side bar charts
fig, axes = plt.subplots(1, 2, figsize=(14,6))

sns.barplot(data=top_emitters, x="CO2 per capita", y="Country", ax=axes[0], color="red")
axes[0].set_title("Top 10 CO2 Emitters per Capita (2022)")

sns.barplot(data=top_renewables, x="Renewable %", y="Country", ax=axes[1], color="green")
axes[1].set_title("Top 10 Renewable Energy Leaders (2022)")

plt.tight_layout()
plt.show()

