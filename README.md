# AI Chess Review and Performance Analysis System

## 1. Project Overview

This project develops a Python-based AI Chess Review and Performance Analysis System using machine learning.

It analyses chess games in PGN format, evaluates individual moves using the Stockfish chess engine, classifies move quality, and trains machine learning models to predict move classifications.

## 2. Objectives

- Analyse chess games using Stockfish.
- Identify Best moves, Good moves, Inaccuracies, Mistakes, and Blunders.
- Prepare a structured chess move dataset.
- Train Decision Tree and Random Forest classifiers.
- Compare model performance using evaluation metrics.
- Predict chess move quality using trained models.

## 3. Technologies Used

- Python
- Google Colab
- Stockfish Chess Engine
- python-chess
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## 4. Dataset

The dataset contains 3,086 move records extracted from 50 chess games.

It includes:
- Game information
- Player information
- Played move and best move
- Evaluation before and after the move
- Centipawn loss
- Move classification

The dataset contains five classes:
- Best
- Good
- Inaccuracy
- Mistake
- Blunder

## 5. Machine Learning Models

Two supervised machine learning classifiers were trained:

### Decision Tree Classifier
- Maximum depth: 5
- Class weighting: Balanced
- Random state: 42

### Random Forest Classifier
- Number of estimators: 200
- Maximum depth: 8
- Class weighting: Balanced
- Random state: 42

## 6. Model Results

| Model | Accuracy |
|---|---:|
| Decision Tree | 47.17% |
| Random Forest | 63.99% |

Random Forest achieved better accuracy on the test dataset.

## 7. Feature Importance

Random Forest feature importance:

| Feature | Importance |
|---|---:|
| Evaluation after move | 46.82% |
| Evaluation before move | 40.91% |
| Move number | 10.65% |
| Player | 1.62% |

## 8. Project Structure

```text
Chess-Review-ML-Project/
├── README.md
├── requirements.txt
├── dataset/
│   └── chess_dataset.csv
├── models/
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── model_features.pkl
├── src/
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
├── results/
│   ├── model_comparison.csv
│   ├── classification_reports.txt
│   └── confusion_matrices.png
└── notebooks/
    └── Chess_Review_Bot.ipynb
```

## 9. Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 10. How to Run

Run these commands from the main project directory.

Train the models:

```bash
python src/train_models.py
```

Evaluate the models:

```bash
python src/evaluate_models.py
```

Predict move quality:

```bash
python src/predict.py
```

## 11. Limitations

- The dataset contains only 50 chess games.
- Model predictions depend on Stockfish-derived evaluation data.
- The machine learning models learn patterns from labels generated using simplified centipawn-loss thresholds.
- Predictions should be treated as estimates, not independent chess-engine analysis.

## 12. Conclusion

The project demonstrates how chess game data can be processed and used to train machine learning models for move-quality classification.

Among the two models, Random Forest achieved better test accuracy than Decision Tree.

Stockfish provides the chess-engine analysis, while the machine learning models learn to predict move classifications from selected evaluation features.