
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import load_data

# Load data
df = load_data("data/sample_data.csv")

# Split features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "outputs/model.pkl")

# SHAP explainability
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.plots.bar(shap_values, show=False)
plt.savefig("outputs/feature_importance.png")
