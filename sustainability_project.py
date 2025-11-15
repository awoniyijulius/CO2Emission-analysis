# Julius' Sustainability Project: CO2 vs Renewable Energy (World Bank Data)
# Indicators:
# EN.ATM.CO2E.PC  -> CO2 emissions (metric tons per capita)
# EG.FEC.RNEW.ZS  -> Renewable energy consumption (% of total final energy)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests, zipfile, io

# 🔧 Constants
CO2_URL = "https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?downloadformat=csv"
RENEW_URL = "https://api.worldbank.org/v2/en/indicator/EG.FEC.RNEW.ZS?downloadformat=csv"
YEAR = "2022"

# 📥 Function to download and extract World Bank data
def download_worldbank_data(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # Grab the first CSV file inside the ZIP (names change over time)
    file_name = [f for f in z.namelist() if f.endswith(".csv")][0]
    df = pd.read_csv(z.open(file_name), skiprows=4)
    return df

# Download datasets
co2 = download_worldbank_data(CO2_URL)
renew = download_worldbank_data(RENEW_URL)

# 🧹 Clean column names
co2.rename(columns={"Country Name": "Country"}, inplace=True)
renew.rename(columns={"Country Name": "Country"}, inplace=True)

# 📊 Select relevant year
co2_data = co2[["Country", YEAR]]
renew_data = renew[["Country", YEAR]]

# 🔗 Merge datasets
merged = pd.merge(co2_data, renew_data, on="Country", how="inner")
merged.columns = ["Country", "CO2 per capita", "Renewable %"]

# 🧠 Correlation
print("\n--- Correlation Matrix ---")
print(merged.corr())

# 🌍 Add regional grouping (expand as needed)
region_map = {
    "Nigeria": "Africa", "Ghana": "Africa", "Kenya": "Africa",
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe",
    "China": "Asia", "India": "Asia", "Japan": "Asia",
    "Brazil": "South America", "Argentina": "South America",
    "United States": "North America", "Canada": "North America",
    "Australia": "Oceania"
}
merged["Region"] = merged["Country"].map(region_map)

# 📊 Compute regional averages
regional_avg = merged.groupby("Region")[["CO2 per capita", "Renewable %"]].mean().reset_index()
regional_avg["Renewable %"] = pd.to_numeric(regional_avg["Renewable %"], errors="coerce")
regional_avg["CO2 per capita"] = pd.to_numeric(regional_avg["CO2 per capita"], errors="coerce")
print("\n--- Regional Averages ---")
print(regional_avg)

# 🎨 Seaborn styling
sns.set_theme(style="whitegrid", palette="muted")

plt.figure(figsize=(12,7))

# Scatter plot for countries
sns.scatterplot(data=merged, x="Renewable %", y="CO2 per capita",
                hue="Region", style="Region", s=80, edgecolor="black", alpha=0.7)

# Overlay regional averages as red stars
plt.scatter(regional_avg["Renewable %"], regional_avg["CO2 per capita"],
            color="red", marker="*", s=300, label="Regional Average")

# Add regression line (trendline)
sns.regplot(data=merged, x="Renewable %", y="CO2 per capita",
            scatter=False, color="blue", line_kws={"linewidth":2, "linestyle":"--"})

# Labels and title
plt.xlabel("Renewable energy consumption (%)", fontsize=12)
plt.ylabel("CO2 emissions per capita (metric tons)", fontsize=12)
plt.title("Renewables vs CO2 Emissions (World Bank, 2022)", fontsize=16)

# Annotate regions (skip NaN values)
for i in regional_avg.index:
    x = regional_avg.loc[i, "Renewable %"]
    y = regional_avg.loc[i, "CO2 per capita"]
    if pd.isna(x) or pd.isna(y):
        continue
    plt.text(float(x) + 0.5,
             float(y) + 0.2,
             regional_avg.loc[i, "Region"], fontsize=10, color="red")

plt.legend()
plt.tight_layout()

# Save before showing
plt.savefig("Renewables_vs_CO2_2022.png", dpi=300)
plt.show()

# 📝 Save outputs
merged.to_csv("Renewables_vs_CO2_2022.csv", index=False)
regional_avg.to_csv("Regional_Averages_2022.csv", index=False)
print("\nData and plot have been saved successfully.")