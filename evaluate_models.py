
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "dataset/chess_dataset.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset and models
df = pd.read_csv(DATA_PATH)

dt_model = joblib.load(f"{MODEL_DIR}/decision_tree_model.pkl")
rf_model = joblib.load(f"{MODEL_DIR}/random_forest_model.pkl")

# Prepare features
df["player_encoded"] = df["player"].map({
    "White": 0,
    "Black": 1
})

features = joblib.load(f"{MODEL_DIR}/model_features.pkl")

X = df[features]
y = df["classification"]

# Recreate the original test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["game_id"])
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Generate predictions
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Accuracy comparison
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ]
})

comparison.to_csv(
    f"{RESULTS_DIR}/model_comparison.csv",
    index=False
)

# Classification reports
with open(
    f"{RESULTS_DIR}/classification_reports.txt",
    "w"
) as file:
    file.write("DECISION TREE REPORT\n")
    file.write(classification_report(y_test, dt_pred))
    file.write("\n\nRANDOM FOREST REPORT\n")
    file.write(classification_report(y_test, rf_pred))

# Confusion matrices
labels = ["Best", "Good", "Inaccuracy", "Mistake", "Blunder"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, predictions, title in zip(
    axes,
    [dt_pred, rf_pred],
    ["Decision Tree", "Random Forest"]
):
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(
    f"{RESULTS_DIR}/confusion_matrices.png",
    dpi=300
)
plt.close()

print(comparison)
print("\nEvaluation results saved in the results folder.")
