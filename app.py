from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
import os

app = FastAPI()

# Define the get_points_map_dict function
def get_points_map_dict(scorecards):
    # Initialize the dictionary
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristic
        current_data = (scorecards[scorecards['Characteristic'] == char]
                        [['Attribute', 'Points']])  # Filter based on characteristic, Then select the attribute & WOE

        # Get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']

            if attribute == 'Missing':
                points_map_dict['Missing'][char] = points
            else:
                points_map_dict[char][attribute] = points
                points_map_dict['Missing'][char] = np.nan

    return points_map_dict

# Define the transform_points function
def transform_points(raw_data, points_map_dict, num_cols):
    points_data = raw_data.copy()

    # Map the data
    for col in points_data.columns:
        map_col = col  # No need to append '_bin' here
        points_data[col] = points_data[col].map(points_map_dict[map_col])

    # Map the data if there is a missing value or out of range value
    for col in points_data.columns:
        map_col = col  # No need to append '_bin' here
        points_data[col] = points_data[col].fillna(value=points_map_dict['Missing'][map_col])

    return points_data

# Define the predict_score function with custom score categorization
def predict_score(raw_data, points_map_dict, num_columns):
    points_data = transform_points(raw_data=raw_data, points_map_dict=points_map_dict, num_cols=num_columns)
    score = int(points_data.sum(axis=1))
    
    if score < 250:
        recommendation = "Very Poor"
    elif score >= 250 and score < 300:
        recommendation = "Poor"
    elif score >= 300 and score < 400:
        recommendation = "Fair"
    elif score >= 400 and score < 500:
        recommendation = "Good"
    elif score >= 500 and score < 600:
        recommendation = "Very Good"
    elif score >= 600 and score < 700:
        recommendation = "Exceptional"
    else:
        recommendation = "Excellent"
        
    return score, recommendation

# Load your scorecards.pkl file and create points_map_dict here
scorecards = pd.read_pickle('scorecards.pkl')
points_map_dict = get_points_map_dict(scorecards=scorecards)

class InputData(BaseModel):
    person_age_bin: int
    person_income_bin: int
    person_emp_length_bin: int
    loan_amnt_bin: int
    loan_int_rate_bin: int
    loan_percent_income_bin: float
    cb_person_cred_hist_length_bin: int
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str

@app.post("/predict_score")
def predict_credit_score(input_data: InputData):
    input_dict = input_data.dict()
    input_table = pd.DataFrame(input_dict, index=[0])
    num_columns = [
        'person_age_bin',
        'person_income_bin',
        'person_emp_length_bin',
        'loan_amnt_bin',
        'loan_int_rate_bin',
        'loan_percent_income_bin',
        'cb_person_cred_hist_length_bin'
    ]
    score, recommendation = predict_score(
        raw_data=input_table,
        points_map_dict=points_map_dict,
        num_columns=num_columns
    )
    return {"Credit Score": recommendation}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
