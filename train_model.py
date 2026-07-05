# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs("models", exist_ok=True)

# Generate synthetic training data
np.random.seed(42)
n = 1000

industries = ['Technology', 'Energy', 'Healthcare', 'Finance', 'Consumer Goods',
              'Industrials', 'Materials', 'Utilities', 'Real Estate', 'Telecom']
regions    = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']

df = pd.DataFrame({
    'Industry'          : np.random.choice(industries, n),
    'Region'            : np.random.choice(regions, n),
    'Year'              : np.random.randint(2015, 2025, n),
    'Revenue'           : np.random.uniform(10, 500000, n),
    'ProfitMargin'      : np.random.uniform(-50, 60, n),
    'MarketCap'         : np.random.uniform(10, 3000000, n),
    'GrowthRate'        : np.random.uniform(-30, 100, n),
    'ESG_Environmental' : np.random.uniform(0, 100, n),
    'ESG_Social'        : np.random.uniform(0, 100, n),
    'ESG_Governance'    : np.random.uniform(0, 100, n),
    'CarbonEmissions'   : np.random.uniform(100, 10000000, n),
    'WaterUsage'        : np.random.uniform(100, 50000000, n),
    'EnergyConsumption' : np.random.uniform(100, 1000000000, n),
})

# Target: weighted average of ESG sub-scores + noise
df['ESG_Overall'] = (
    0.35 * df['ESG_Environmental'] +
    0.35 * df['ESG_Social'] +
    0.30 * df['ESG_Governance'] +
    np.random.normal(0, 3, n)
).clip(0, 100)

# Encode categorical columns
le_industry = LabelEncoder().fit(df['Industry'])
le_region   = LabelEncoder().fit(df['Region'])

df['Industry'] = le_industry.transform(df['Industry'])
df['Region']   = le_region.transform(df['Region'])

features = [col for col in df.columns if col != 'ESG_Overall']
X = df[features]
y = df['ESG_Overall']

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0)
model.fit(X_scaled, y)


# Save everything
joblib.dump(model,       "models/esg_ridge_model.pkl")
joblib.dump(scaler,      "models/esg_scaler.pkl")
joblib.dump(le_industry, "models/le_industry.pkl")
joblib.dump(le_region,   "models/le_region.pkl")
joblib.dump(features,    "models/feature_names.pkl")

print("✅ All model files saved to models/ folder!")