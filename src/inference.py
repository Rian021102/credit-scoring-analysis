import numpy as np
import pickle
import pandas as pd

def get_points_map_dict(scorecards):
    # Initialize the dictionary
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
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

# Function to predict the credit score
def predict_score(raw_data, cutoff_score, points_map_dict, num_columns):

    # Transform raw input values into score points
    points = transform_points(raw_data = raw_data,
                              points_map_dict = points_map_dict,
                              num_cols = num_columns)

    # Caculate the score as the total points
    score = int(points.sum(axis=1))

    print(f"Credit Score : ", score)

    if score > cutoff_score:
        print("Recommendation : APPROVE")
    else:
        print("Recommendation : REJECT")

    return score

def main():
    num_columns = ['person_age_bin',
                   'person_income_bin',
                   'person_emp_length_bin',
                   'loan_amnt_bin',
                   'loan_int_rate_bin',
                   'loan_percent_income_bin',
                   'cb_person_cred_hist_length_bin']

    # import scorecards.pkl
    scorecards = pd.read_pickle('/Users/rianrachmanto/pypro/project/credit-scoring-analysis/model/scorecards.pkl')
    points_map_dict = get_points_map_dict(scorecards=scorecards)
    print(points_map_dict)

    input = {
        'person_age_bin': 50,
        'person_income_bin': 15000,
        'person_emp_length_bin': 2,
        'loan_amnt_bin': 9000,
        'loan_int_rate_bin': 8,
        'loan_percent_income_bin': 0.21,
        'cb_person_cred_hist_length_bin': 4,
        'person_home_ownership': 'RENT',
        'loan_intent': 'MEDICAL',
        'loan_grade': 'D',
        'cb_person_default_on_file': 'Y'
    }
    input_table = pd.DataFrame(input, index=[0])
    input_points = transform_points(raw_data=input_table,
                                    points_map_dict=points_map_dict,
                                    num_cols=num_columns)

    print(input_points)

    input_score = predict_score(raw_data = input_table,cutoff_score = 150,points_map_dict = points_map_dict,num_columns = num_columns)



if __name__ == "__main__":
    main()
