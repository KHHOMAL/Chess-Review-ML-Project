
import joblib
import pandas as pd

MODEL_DIR = "models"

# Load trained models
dt_model = joblib.load(
    f"{MODEL_DIR}/decision_tree_model.pkl"
)

rf_model = joblib.load(
    f"{MODEL_DIR}/random_forest_model.pkl"
)

features = joblib.load(
    f"{MODEL_DIR}/model_features.pkl"
)

print("CHESS MOVE QUALITY PREDICTOR")
print("-" * 35)

# Get input from user
move_number = int(input("Enter move number: "))
eval_before = float(input("Enter evaluation before move (centipawns): "))
eval_after = float(input("Enter evaluation after move (centipawns): "))

player = input("Enter player (White/Black): ").strip().capitalize()

if player not in ["White", "Black"]:
    raise ValueError("Player must be White or Black.")

player_encoded = 0 if player == "White" else 1

# Prepare input
input_data = pd.DataFrame([{
    "move_number": move_number,
    "eval_before": eval_before,
    "eval_after": eval_after,
    "player_encoded": player_encoded
}])

input_data = input_data[features]

# Predict using both models
dt_prediction = dt_model.predict(input_data)[0]
rf_prediction = rf_model.predict(input_data)[0]

print("\nPREDICTION RESULTS")
print("-" * 35)
print("Decision Tree:", dt_prediction)
print("Random Forest:", rf_prediction)
