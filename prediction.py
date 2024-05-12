import joblib
import pandas as pd
 

def get_user_input():
    input_data = {}
    input_data['Age'] = int(input("Age : "))
    input_data['JobLevel'] = int(input("Job Level: "))
    input_data['Education'] = int(input("Education: "))
    return input_data

def make_prediction(input_data):
    model = joblib.load("logistic_regression_model.joblib")
    feature_names = ['Age', 'JobLevel', 'Education']
    input_df = pd.DataFrame([input_data], columns=feature_names)

    prediction = model.predict(input_df)
    return prediction[0]

user_input = get_user_input()
prediction_result = make_prediction(user_input)
print("=====================================")
print("Predicted Attrition:", "Yes or 1.0" if prediction_result == 1 else "No or 0.0")
print("=====================================")