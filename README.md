# In-Vehicle-Coupon-Recommendation-Using-ML-models
**Project Overview**

This project applies machine learning techniques to predict customer behavior using a structured dataset. The dataset contains categorical and numerical features, and the goal is to classify whether a customer will respond positively (Y=1) or negatively (Y=0) to an offer.

**Dataset**

The dataset used in this project is stored in a CSV file (Team3_Dataset.csv). It contains various demographic and behavioral attributes of customers. Missing values, outliers, and categorical variables were handled before applying machine learning models.

**Data Preprocessing**

Handling Missing Values:

Replaced empty strings with NA.

Visualized missing values using bar plots and heatmaps.

No imputation was performed as categorical features were analyzed directly.

Feature Selection and Transformation:

Dropped irrelevant and low-variance columns (car, toCoupon_GEQ5min, temperature).

Grouped low-frequency occupation categories into broader classes.

Converted categorical variables into dummy variables using one-hot encoding.

Removed highly correlated features (correlation > 0.8).

**Machine Learning Models**

Three classification models were trained and evaluated:

Logistic Regression:

Trained using forward, backward, and stepwise selection methods.

Evaluated at different probability cutoffs (0.3, 0.4, 0.5) using accuracy, sensitivity, and specificity metrics.

K-Nearest Neighbors (KNN):

Applied cross-validation for hyperparameter tuning.

Evaluated using probability cutoffs.

Classification and Regression Tree (CART):

Built a decision tree model and applied pruning to improve performance.

Compared minimum error tree and best-pruned tree using accuracy scores.

**Model Evaluation****

Confusion Matrices and performance metrics (Accuracy, Sensitivity, Specificity) were computed for each model.

Cutoff Threshold Analysis helped determine optimal classification thresholds.

**Visualization:**

Decision trees were plotted using rpart.plot.

Missing value patterns and distributions were visualized.

**Results**

Logistic regression, KNN, and CART models provided insights into customer behavior patterns.

Decision trees and logistic regression feature importance analyses helped interpret key drivers of customer responses.

The best-performing model can be selected based on business objectives (e.g., prioritizing sensitivity over specificity).
