import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from codecarbon import EmissionsTracker

def train_models(data):
    results = []
    X = data.drop(columns=["date", "Appliances", "lights"])
    y = data["Appliances"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    }

    for name, model in models.items():
        tracker = EmissionsTracker(output_file=f"{name}_emissions.csv")
        tracker.start()

        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()

        emissions = tracker.stop()

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "R2 Score": round(r2, 2),
            "Training Time (s)": round(end - start, 2),
            "CO2 Emissions (kg)": round(emissions, 4)
        })

    return pd.DataFrame(results)
