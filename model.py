import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


DATA_PATH = r"C:\Users\Admin\Desktop\flask_ml_app\Housing_Price.xlsx"
MODEL_PATH = "model.pkl"
SELECTOR_PATH = "selector.pkl"
SCALER_PATH = "scaler.pkl"


def train_model():
    df = pd.read_excel("dataset/Housing_Price.xlsx")






    # Fill missing values with median
    df.fillna(df.median(), inplace=True)

    # Define features and target
    feature_columns = df.columns[:5]  # First 5 columns as features
    target_column = df.columns[-1]    # Last column as target

    X = df[feature_columns]  
    y = df[target_column]    

    # Feature Selection - Keep the top 3 best features
    selector = SelectKBest(score_func=f_regression, k=3)
    X_selected = selector.fit_transform(X, y)

    # Save selected feature indices
    with open(SELECTOR_PATH, "wb") as f:
        pickle.dump(selector, f)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Save the scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define model and hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
    }

    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model selection
    best_model = grid_search.best_estimator_

    # Evaluate model performance
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Best Model R² Score: {r2:.4f}")

    # Save the best model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print("✅ Optimized model trained & saved as 'model.pkl'")

def predict_price(features):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SELECTOR_PATH, "rb") as f:
        selector = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Convert input to numpy array
    features = np.array(features).reshape(1, -1)

    # Select only the relevant features
    features_selected = selector.transform(features)

    # Apply the same scaling as during training
    features_scaled = scaler.transform(features_selected)

    predicted_price = model.predict(features_scaled)[0]
    
    return round(predicted_price, 2)

if __name__ == "__main__":
    train_model()