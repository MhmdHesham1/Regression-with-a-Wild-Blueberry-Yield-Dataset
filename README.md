Regression with Wild Blueberry Yield Dataset

This project involves using various regression models to predict the yield of wild blueberries based on different features. The models are trained, evaluated, and compared using several machine learning algorithms, including Random Forest, Gradient Boosting, Decision Tree, K-Nearest Neighbors, XGBoost, and CatBoost. A Voting Regressor is also employed to combine the predictions of individual models.
Table of Contents

    Project Overview
    Data
    Models Used
    Evaluation Metrics
    Installation
    Usage
    Results
    Contributing
    License

Project Overview

The goal of this project is to develop machine learning models that can accurately predict the yield of wild blueberries based on a set of input features. Various regression algorithms are utilized to achieve this, and their performance is evaluated and compared.
Data

The dataset used in this project contains features related to wild blueberry yield. It includes variables like environmental conditions, soil characteristics, and more. The dataset is divided into training and testing sets for model evaluation.
Models Used

The following machine learning models are used in this project:

    Random Forest Regressor
    Gradient Boosting Regressor
    Decision Tree Regressor
    K-Nearest Neighbors Regressor
    XGBoost Regressor
    CatBoost Regressor
    Voting Regressor (combines the above models)

Evaluation Metrics

The models are evaluated using the following metrics:

    Mean Squared Error (MSE)
    Mean Absolute Error (MAE)
    R² Score

Installation

To run this project, you need to have Python installed along with several libraries. You can install the necessary dependencies using the following command:
pip install pandas numpy scikit-learn xgboost catboost

Usage

    Train the Models: The models are trained on the training dataset (X_train_scaled and y_train).
    Evaluate the Models: Each model is evaluated on the test dataset (X_test_scaled and y_test) using the specified evaluation metrics.
    Make Predictions: Use the trained models to make predictions on new data (testdata).
    Create Submission File: The predictions are saved in a submission.csv file for submission or further analysis.

Example Code
# Train and evaluate a Random Forest Regressor
RandomForestModel = RandomForestRegressor(random_state=42)
RandomForestModel.fit(X_train_scaled, y_train)
RF_y_pred = RandomForestModel.predict(X_test_scaled)
rf_results = evaluate_model(y_test, RF_y_pred, "Random Forest")

# Make predictions on new data with CatBoost Regressor
test_predictions = CatBoostModel.predict(testdata)
submission = pd.DataFrame({'id': testdata['id'], 'yield': test_predictions})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")


Results

The results of each model are displayed in terms of MSE, MAE, and R² score. A Voting Regressor that combines all models provides an ensemble prediction, which often improves overall performance.
Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss changes.
License

This project is licensed under the MIT License - see the LICENSE file for details.
