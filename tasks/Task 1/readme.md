## Task 1 – Baseline Models and Comparison

### EDA
- Target distribution shows skew in median house value.
- Correlation heatmap highlights strongest predictors (e.g., MedInc, HouseAge).
- Scatter plots confirm non-linear relationships.

### Feature Engineering
- Added: rooms_per_household, bedrooms_per_room, income_x_age, log_population.
- These capture ratios, interactions, and reduce skew.

### Model Results
| Model                  | MAE   | RMSE  | R²   |
|-------------------------|-------|-------|------|
| Ridge (tuned)           | ...   | ...   | ...  |
| RandomForest (tuned)    | ...   | ...   | ...  |
| XGBRegressor (tuned)    | ...   | ...   | ...  |

### Conclusion
- **Best model by R²**: XGBRegressor (captures non-linearities).  
- Ridge is interpretable but underfits.  
- RandomForest performs well but less efficient.  
- XGB balances accuracy and generalization, making it the strongest choice.
