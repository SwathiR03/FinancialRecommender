import pandas as pd
import math
import numpy as np

# Load your dataset here
data = pd.read_csv("Invest-SW-with-Data.csv")
# Define features and target attribute
features = [feat for feat in data.columns if feat not in ["InvestAmt_Percent","Insurance_Percent","Equity_Percent","MF_Percent","Debt_Percent","GoldSilver_Percent","RealEstate_Percent","Crypto_Percent"]]
target_attribute = "Equity_Percent"

# Define a class for the regression tree node
class RegressionNode:
    def __init__(self):
        self.children = []
        self.feature = ""
        self.split_value = None
        self.prediction = None

# Update the splitting criterion and tree construction for regression
def mse_loss(examples):
    actual_values = examples[target_attribute]
    mean_value = actual_values.mean()
    mse = ((actual_values - mean_value) ** 2).mean()
    return mse

def best_split(examples, attributes):
    best_mse = float("inf")
    best_feature = None
    best_split_value = None

    for feature in attributes:
        unique_values = examples[feature].unique()
        for value in unique_values:
            left_subset = examples[examples[feature] <= value]
            right_subset = examples[examples[feature] > value]
            mse = mse_loss(left_subset) + mse_loss(right_subset)

            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_split_value = value

    return best_feature, best_split_value

def build_regression_tree(examples, attributes):
    node = RegressionNode()

    if len(attributes) == 0:
        node.prediction = examples[target_attribute].mean()
        return node

    if mse_loss(examples) == 0.0:
        node.prediction = examples[target_attribute].mean()
        return node

    best_feature, best_split_value = best_split(examples, attributes)
    node.feature = best_feature
    node.split_value = best_split_value
    

    left_subset = examples[examples[best_feature] <= best_split_value]
    right_subset = examples[examples[best_feature] > best_split_value]

    if len(left_subset) > 0:
        node.children.append(build_regression_tree(left_subset, attributes))
    if len(right_subset) > 0:
        node.children.append(build_regression_tree(right_subset, attributes))

    return node

# Update the prediction function for regression
def predict_regression(node, example):
    if node.prediction is not None:
        return node.prediction
    if example[node.feature] <= node.split_value:
        return predict_regression(node.children[0], example)
    else:
        return predict_regression(node.children[1], example)

# Build the regression tree for target attribute = InvestAmt_Percent
target_attribute = "InvestAmt_Percent"
root_Invest = build_regression_tree(data, features)

#Build RT for Insurance_Percent
target_attribute = "Insurance_Percent"
root_Insurance = build_regression_tree(data, features)

# Build RT for Equity_Percent
target_attribute = "Equity_Percent"
root_Equity= build_regression_tree(data, features)

# Build RT for MF_Percent
target_attribute = "MF_Percent"
root_MF= build_regression_tree(data, features)

# Build RT for Debt_Percent
target_attribute = "Debt_Percent"
root_Debt= build_regression_tree(data, features)

# Build RT for GoldSilver_Percent
target_attribute = "GoldSilver_Percent"
root_GoldSilver= build_regression_tree(data, features)

# Build RT for RealEstate_Percent
target_attribute = "RealEstate_Percent"
root_RealEstate= build_regression_tree(data, features)

# Build RT for Crypto_Percent
target_attribute = "Crypto_Percent"
root_Crypto= build_regression_tree(data, features)

def get_user_input():
    Age_Var = input("Enter Age (e.g., '36 - 60'): ")
    Monthly_Income_Var = input("Enter Monthly Income (e.g., '0-50000'): ")
    Occupation_Var = input("Enter Occupation (e.g., 'Corporate Salaried'): ")
    FamilySize_Var = input("Enter Family Size (e.g., '5 to 6'): ")
    RiskProfile_Var = input("Enter Risk Profile (e.g., 'Conservative'): ")
    
    new_example = {
        "Age": Age_Var,
        "MonthlyIncome": Monthly_Income_Var,
        "Occupation": Occupation_Var,
        "FamilySize": FamilySize_Var,
        "RiskProfile": RiskProfile_Var
    }
    
    return new_example

# Allow the user to enter input and make predictions
while True:
    
    print("Enter input values for prediction (or 'exit' to quit):")
    user_input = input()
    
    if user_input.lower() == "exit":
        break
    
    new_example = get_user_input()
    
    # Make predictions for different target attributes
    prediction_Invest = predict_regression(root_Invest, new_example)
    print("Predicted Value of Invest Amt %:", prediction_Invest)
    prediction_Insurance = predict_regression(root_Insurance, new_example)
    print("Predicted Value of Insurance Amt %:", prediction_Insurance)
    prediction_Equity = predict_regression(root_Equity, new_example)
    print("Predicted Value of Equity Amt %:", prediction_Equity)
    prediction_MF = predict_regression(root_MF, new_example)
    print("Predicted Value of MF Amt %:", prediction_MF)
    prediction_Debt = predict_regression(root_Debt, new_example)
    print("Predicted Value of Debt Amt %:", prediction_Debt)
    prediction_GoldSilver = predict_regression(root_GoldSilver, new_example)
    print("Predicted Value of GoldSilver Amt %:", prediction_GoldSilver)
    prediction_RealEstate = predict_regression(root_RealEstate, new_example)
    print("Predicted Value of RealEstate Amt %:", prediction_RealEstate)
    prediction_Crypto = predict_regression(root_Crypto, new_example)
    print("Predicted Value of Crypto Amt %:", prediction_Crypto)