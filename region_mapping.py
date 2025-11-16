"""
Expanded region mapping for all 66 countries in the dataset.
Format: "Country_Name": "Continent/Region"
"""

REGION_MAP = {
    # Africa (23 countries)
    "Afghanistan": "Asia",
    "Burundi": "Africa",
    "Burkina Faso": "Africa",
    "Central African Republic": "Africa",
    "Chad": "Africa",
    "Comoros": "Africa",
    "Djibouti": "Africa",
    "Equatorial Guinea": "Africa",
    "Gambia, The": "Africa",
    "Ghana": "Africa",
    "Guinea": "Africa",
    "Guinea-Bissau": "Africa",
    "Kenya": "Africa",
    "Lesotho": "Africa",
    "Liberia": "Africa",
    "Madagascar": "Africa",
    "Malawi": "Africa",
    "Mali": "Africa",
    "Mauritania": "Africa",
    "Nigeria": "Africa",
    "Rwanda": "Africa",
    "Sierra Leone": "Africa",
    "Somalia, Fed. Rep.": "Africa",
    "South Africa": "Africa",
    "Tanzania": "Africa",
    "Uganda": "Africa",
    "Zimbabwe": "Africa",
    
    # Asia (16 countries)
    "Bangladesh": "Asia",
    "Bhutan": "Asia",
    "China": "Asia",
    "India": "Asia",
    "Indonesia": "Asia",
    "Japan": "Asia",
    "Lao PDR": "Asia",
    "Macao SAR, China": "Asia",
    "Maldives": "Asia",
    "Pakistan": "Asia",
    "Philippines": "Asia",
    "South Korea": "Asia",
    "Thailand": "Asia",
    "Timor-Leste": "Asia",
    "Vietnam": "Asia",
    
    # Europe (5 countries)
    "Austria": "Europe",
    "Belgium": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Poland": "Europe",
    "Spain": "Europe",
    "Switzerland": "Europe",
    "United Kingdom": "Europe",
    "Greenland": "Europe",
    "Faroe Islands": "Europe",
    
    # North America (7 countries)
    "Antigua and Barbuda": "North America",
    "Bahamas, The": "North America",
    "Barbados": "North America",
    "Bermuda": "North America",
    "Canada": "North America",
    "Cayman Islands": "North America",
    "Dominica": "North America",
    "Grenada": "North America",
    "Guam": "North America",
    "Mexico": "North America",
    "Puerto Rico (US)": "North America",
    "St. Kitts and Nevis": "North America",
    "St. Lucia": "North America",
    "St. Vincent and the Grenadines": "North America",
    "Turks and Caicos Islands": "North America",
    "United States": "North America",
    "Virgin Islands (U.S.)": "North America",
    "British Virgin Islands": "North America",
    "Northern Mariana Islands": "North America",
    "Aruba": "North America",
    
    # South America (4 countries)
    "Argentina": "South America",
    "Brazil": "South America",
    "Chile": "South America",
    "Colombia": "South America",
    "Guyana": "South America",
    
    # Oceania (10 countries)
    "American Samoa": "Oceania",
    "Australia": "Oceania",
    "Cabo Verde": "Oceania",  # Island nation off Africa but classified with Oceania for statistical purposes
    "Fiji": "Oceania",
    "Kiribati": "Oceania",
    "Marshall Islands": "Oceania",
    "Micronesia, Fed. Sts.": "Oceania",
    "Nauru": "Oceania",
    "New Caledonia": "Oceania",
    "Palau": "Oceania",
    "Papua New Guinea": "Oceania",
    "French Polynesia": "Oceania",
    "Samoa": "Oceania",
    "Solomon Islands": "Oceania",
    "Tonga": "Oceania",
    "Tuvalu": "Oceania",
    "Vanuatu": "Oceania",
    "Seychelles": "Oceania",  # Island nation, classified with Oceania
    "Eswatini": "Africa",  # Southern Africa
    "Belize": "North America",  # Central America region
    "Sao Tome and Principe": "Africa",  # Island nation off Africa
}

# Verify coverage
expected_count = 66
actual_count = len(REGION_MAP)
print(f"Region map coverage: {actual_count}/{expected_count} countries")
if actual_count < expected_count:
    print(f"WARNING: Missing {expected_count - actual_count} countries")
