# NFL Elo Model with Machine Learning-Adjusted QB Ratings

This project builds a predictive Elo rating system for NFL teams that incorporates margin of victory, quarterback performance, and machine learning-based adjustments. The model is trained on historical game data and optimized to predict point spreads and betting outcomes.

## Project Goals

- Build an Elo-based model for NFL teams using score differentials
- Integrate quarterback performance into team ratings using machine learning
- Tune model parameters via time-series cross-validation
- Predict game outcomes and evaluate model performance using log loss and betting return

## Data

- Source: [nflfastR](https://github.com/nflverse/nflfastR) (data not included in repo)
- Games from 2000 to 2024 (train: 2000–2019, test: 2020–2024)
- Data includes game results, point spreads, team names, and starting quarterbacks

## Tech Stack

- Python
- `pandas`, `numpy`, `scikit-learn`, `optuna`
- Elo rating logic and parameter tuning implemented from scratch

## Modeling Approach

- Elo rating system updated by a multiplicative margin-of-victory factor
- Separate ratings for quarterbacks and teams; QB influence decays over time
- Machine learning pipeline trained on past results to fine-tune QB weight, K-factor, and home field adjustment
- Cross-validation via rolling windows to prevent lookahead bias

## Evaluation

- **Metrics**:
  - Mean Squared Error (MSE) of predicted vs. actual point spread
  - Log loss for classification (win probability)
  - Simulated betting return using Kelly criterion

- **Results**:
  - Model captures QB impact better than baseline Elo
  - Outperforms Vegas spread on selected high-confidence bets
  - Optimized parameters yield consistent improvements across years

## File Structure

- `nfl_elo_ML.py`: Main script for training, evaluating, and simulating the model
- `data/`: Placeholder for game and player data (not included for size reasons)
- `results/`: Saved predictions and performance metrics
