# SpaceshipSurvival
This project is focused on predicting the survival of passengers aboard the Spaceship Titanic using machine learning techniques. The dataset provides information about passengers on a spaceship journey, including their demographic details and whether they survived or not. The objective is to predict survival based on features like age, gender, family size, and more.
Table of Contents

    Overview
    Dataset
    Data Preprocessing
    Modeling Techniques
    Evaluation Metrics
    Results
    Installation
    Usage
    License

Overview

In this project, we used machine learning to predict whether a passenger survived or not on the Spaceship Titanic. The dataset is fictional but follows a similar structure to the Titanic dataset, with features such as gender, age, and class. The task is to train a model using historical data, and then classify whether a new passenger would survive or not based on the features.
Dataset

The dataset used in this project is based on the Spaceship Titanic competition on Kaggle. It contains information about passengers and their survival status.
Key Columns in the Dataset:

    PassengerId: Unique identifier for each passenger.
    HomePlanet: The planet of origin.
    CryoSleep: Whether the passenger was in cryosleep (True/False).
    Cabin: Cabin number.
    Destination: The passenger’s destination.
    Age: Age of the passenger.
    VIP: Whether the passenger is a VIP (True/False).
    RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: Expenditures for various services.
    Name: Name of the passenger.
    Survived: Target variable (1 = Survived, 0 = Did not survive).

Data Preprocessing

To prepare the dataset for modeling, we perform several preprocessing steps:

    Handle Missing Values:
    Fill or drop rows with missing data (e.g., Age, Cabin, CryoSleep, Destination).

    Feature Engineering:
        Convert categorical variables such as CryoSleep, VIP, and HomePlanet into numerical values.
        Create new features, such as total expenditure on services (RoomService + FoodCourt + ShoppingMall + Spa + VRDeck).

    Scaling and Normalization:
    Normalize features like Age and expenditures for better model performance.

    Train-Test Split:
    Split the dataset into training and testing sets (e.g., 80%-20%) to evaluate model performance.

Modeling Techniques

The following machine learning models were implemented:

    Logistic Regression
    Random Forest Classifier
    K-Nearest Neighbors (KNN)
    Support Vector Machine (SVM)
    Gradient Boosting Classifier (optional)

These models were trained and evaluated to predict whether a passenger survived or not.
Evaluation Metrics

The models are evaluated based on the following metrics:

    Accuracy: Proportion of correct predictions made by the model.
    Precision: Proportion of true positives out of all positive predictions.
    Recall: Proportion of true positives out of all actual positives.
    F1-Score: Harmonic mean of precision and recall, providing a balanced measure of both.

Results
Example Model Performance (Hypothetical Results):
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.80	0.75	0.78	0.76
Random Forest	0.85	0.83	0.80	0.81
KNN	0.78	0.74	0.72	0.73
SVM	0.82	0.79	0.77	0.78
Installation

To run this project, you will need the required dependencies. You can install them using pip:

pip install -r requirements.txt

The requirements.txt file should contain:

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn

Usage

    Load the Dataset:
    Download the dataset and place it in the project directory. Update the file path in the script if necessary.

    Run the Code:
    Execute the script to preprocess the data, train the models, and evaluate their performance.

    Predict Survival:
    Once the models are trained, you can use them to predict survival for a new passenger:

    new_passenger = [3, 'Earth', True, 'A23', 29, 0, 2000, True]
    prediction = predict_survival(new_passenger)
    print(f"Prediction (1 = Survived, 0 = Did not survive): {prediction}")

    Visualize Results:
    Use matplotlib and seaborn to visualize the model performance and the feature importance.

License

This project is licensed under the MIT License - see the LICENSE file for details.
