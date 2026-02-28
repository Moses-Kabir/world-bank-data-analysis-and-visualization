import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
file_path = r'D:\TRANSPORTATION ENGINEERING\AUTOSAVE\WB_RISE_EE_TRN.csv'
df = pd.read_csv(file_path)

# 2. CLEANING
cols = {'REF_AREA': 'ISO', 'REF_AREA_LABEL': 'Country', 'TIME_PERIOD': 'Year', 'OBS_VALUE': 'Score'}
df_clean = df[list(cols.keys())].copy()
df_clean.rename(columns=cols, inplace=True)
df_clean['Score'] = pd.to_numeric(df_clean['Score'], errors='coerce')
df_clean = df_clean.dropna(subset=['Score'])
latest_data = df_clean[df_clean['Year'] == 2023].copy()

# 3. INTERACTIVE MAP WITH TOP COUNTRY LABELS
# Initialize the map
fig = px.choropleth(
    latest_data,
    locations="ISO",
    color="Score",
    hover_name="Country",
    color_continuous_scale=px.colors.diverging.RdYlGn,
    range_color=[0, 100],
    title="<b>Global Transport Sector Regulatory Scores (2023)</b>"
)

# --- ADD LABELS FOR TOP COUNTRIES ONLY ---
# We only label the top 20 to avoid messy overlap
top_20 = latest_data.nlargest(20, 'Score')

fig.add_trace(go.Scattergeo(
    locations=top_20['ISO'],
    text=top_20['Country'],
    mode='text',
    textposition='top center',
    textfont=dict(size=9, color='black'),
    name='Top 20 Performers',
    showlegend=False
))

# 4. FINAL LAYOUT ADJUSTMENTS
fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True, landcolor="lightgray"),
    coloraxis_colorbar=dict(title="Score (0-100)"),
    margin={"r":0,"t":50,"l":0,"b":0}
)

fig.show()

# 5. SUMMARY TABLE PRINT OUT
print("--- Top 20 Countries for Transport Sector (2023) ---")
print(top_20[['Country', 'Score']].to_string(index=False))