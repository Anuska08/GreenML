# preprocess.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model():
    df = pd.read_csv(r"D:\Greenml\data\KAG_energydata_complete.csv")


    # Drop non-useful columns
    df = df.drop(columns=["date", "lights"])

    # Features & Target
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model & feature names
    with open("model.pkl", "wb") as f:
        pickle.dump((model, X.columns.tolist()), f)

if __name__ == "__main__":
    train_and_save_model()
