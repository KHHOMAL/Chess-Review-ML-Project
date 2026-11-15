
import os
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Project paths
DATA_PATH = "dataset/chess_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare features
df["player_encoded"] = df["player"].map({
    "White": 0,
    "Black": 1
})

features = [
    "move_number",
    "eval_before",
    "eval_after",
    "player_encoded"
]

X = df[features]
y = df["classification"]

# Split by game to prevent data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["game_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Train Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate accuracy
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# Save models and feature names
joblib.dump(dt_model, f"{MODEL_DIR}/decision_tree_model.pkl")
joblib.dump(rf_model, f"{MODEL_DIR}/random_forest_model.pkl")
joblib.dump(features, f"{MODEL_DIR}/model_features.pkl")

print("Models trained and saved successfully!")
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
