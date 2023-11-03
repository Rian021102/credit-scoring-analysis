# Load data manipulation package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn import metrics
#load data

def loaddata(path):
    df=pd.read_csv(path)
    print(df.T)
    response_variable=df['loan_status']
    y=response_variable
    X=df.drop(columns=['loan_status'],axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)
    print(X_train.isnull().sum())
    return X_train,X_test,y_train,y_test,response_variable

def databinning(X_train, y_train,response_variable):
    num_columns = ['person_age',
               'person_income',
               'person_emp_length',
               'loan_amnt',
               'loan_int_rate',
               'loan_percent_income',
               'cb_person_cred_hist_length']


    # Define data with categorical predictors
    cat_columns = ['person_home_ownership',
                    'loan_intent',
                    'loan_grade',
                    'cb_person_default_on_file']
    
    data_train=pd.concat([X_train,y_train],axis=1)
    print(data_train.head())
    print(data_train.isna().sum())
    
    # Create a function for binning the numerical predictor
    def create_binning(data, predictor_label, num_of_bins):
        """
        Function for binning numerical predictor.

        Parameters
        ----------
        data : array like
        The name of dataset.

        predictor_label : object
        The label of predictor variable.

        num_of_bins : integer
        The number of bins.


        Return
        ------
        data : array like
        The name of transformed dataset.

        """
        # Create a new column containing the binned predictor
        data[predictor_label + "_bin"] = pd.qcut(data[predictor_label],
                                                q = num_of_bins)

        return data
    for column in num_columns:
        data_train_binned = create_binning(data = data_train,
                                        predictor_label = column,
                                        num_of_bins = 4)
    print(data_train_binned.T)
    print(data_train_binned.isna().sum())
    # # Define columns with missing values
    # missing_columns = ['person_emp_length_bin',
    #                    'loan_int_rate_bin',
    #                    'cb_person_cred_hist_length_bin']

    # Define columns with missing values
    missing_columns = ['person_emp_length_bin',
                       'loan_int_rate_bin']
    
    # Perform grouping for all columns
    for column in missing_columns:

        # Add category 'Missing' to replace the missing values
        data_train_binned[column] = data_train_binned[column].cat.add_categories('Missing')

        # Replace missing values with category 'Missing'
        data_train_binned[column].fillna(value = 'Missing',
                                        inplace = True)
    print(data_train_binned.T)
    return data_train_binned

def create_crosstablist(data_train_binned,response_variable):
    num_columns = ['person_age',
                   'person_income',
                   'person_emp_length',
                   'loan_amnt',
                   'loan_int_rate',
                   'loan_percent_income',
                   'cb_person_cred_hist_length']


    # Define data with categorical predictors
    cat_columns = ['person_home_ownership',
                   'loan_intent',
                   'loan_grade',
                   'cb_person_default_on_file']

    # Define the initial empty list
    crosstab_num = []

    for column in num_columns:

        # Create a contingency table
        crosstab = pd.crosstab(data_train_binned[column + "_bin"],
                                data_train_binned[response_variable],
                                margins = True)

        # Append to the list
        crosstab_num.append(crosstab)
    
    # Define the initial empty list
    crosstab_cat = []

    for column in cat_columns:

        # Create a contingency table
        crosstab = pd.crosstab(data_train_binned[column],
                                data_train_binned[response_variable],
                                margins = True)

        # Append to the list
        crosstab_cat.append(crosstab)
    
    crosstab_list = crosstab_num + crosstab_cat
    print(crosstab_list)
    return crosstab_list


def create_woe_iv(crosstab_list):
    num_columns = ['person_age',
                   'person_income',
                   'person_emp_length',
                   'loan_amnt',
                   'loan_int_rate',
                   'loan_percent_income',
                   'cb_person_cred_hist_length']


    # Define data with categorical predictors
    cat_columns = ['person_home_ownership',
                   'loan_intent',
                   'loan_grade',
                   'cb_person_default_on_file']
    
    # Define the initial list for WOE
    WOE_list = []

    # Define the initial list for IV
    IV_list = []

    #Create the initial table for IV
    IV_table = pd.DataFrame({'Characteristic': [],
                             'Information Value' : []})

    # Perform the algorithm for all crosstab
    for crosstab in crosstab_list:

    # Calculate % Good
        crosstab['p_good'] = crosstab[0]/crosstab[0]['All']

        # Calculate % Bad
        crosstab['p_bad'] = crosstab[1]/crosstab[1]['All']

        # Calculate the WOE
        crosstab['WOE'] = np.log(crosstab['p_good']/crosstab['p_bad'])

        # Calculate the contribution value for IV
        crosstab['contribution'] = (crosstab['p_good']-crosstab['p_bad'])*crosstab['WOE']

        # Calculate the IV
        IV = crosstab['contribution'][:-1].sum()

        add_IV = {'Characteristic': crosstab.index.name,
                  'Information Value': IV}

        WOE_list.append(crosstab)
        IV_list.append(add_IV)
    print(IV_list)
    print(WOE_list)
    return IV_list,WOE_list

def create_woe_table(crosstab_list,IV_list):
    WOE_table = pd.DataFrame({'Characteristic': [],
                              'Attribute': [],
                              'WOE': []})

    for i in range(len(crosstab_list)):

    # Define crosstab and reset index
        crosstab = crosstab_list[i].reset_index()

        # Save the characteristic name
        char_name = crosstab.columns[0]

        # Only use two columns (Attribute name and its WOE value)
        # Drop the last row (average/total WOE)
        crosstab = crosstab.iloc[:-1, [0,-2]]
        crosstab.columns = ['Attribute', 'WOE']

        # Add the characteristic name in a column
        crosstab['Characteristic'] = char_name

        WOE_table = pd.concat((WOE_table, crosstab),
                              axis = 0)

        # Reorder the column
        WOE_table.columns = ['Characteristic',
                             'Attribute',
                             'WOE']
    # Put all IV in the table
    IV_table = pd.DataFrame(IV_list)
    IV_table
    # Define the predictive power of each characteristic
    strength = []

    # Assign the rule of thumb regarding IV
    for iv in IV_table['Information Value']:
        if iv < 0.02:
            strength.append('Unpredictive')
        elif iv >= 0.02 and iv < 0.1:
            strength.append('Weak')
        elif iv >= 0.1 and iv < 0.3:
            strength.append('Medium')
        elif iv >= 0.3 and iv < 0.5:
            strength.append('Strong')
        else:
            strength.append('Very strong')

    # Assign the strength to each characteristic
    IV_table = IV_table.assign(Strength = strength)

    # Sort the table by the IV values
    IV_table.sort_values(by='Information Value')
    print(WOE_table)
    print(IV_table.sort_values(by='Information Value'))
    return WOE_table,IV_table

# Function to generate the WOE mapping dictionary
def get_woe_map_dict(WOE_table):

    # Initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}

    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'WOE']])                 # Then select the attribute & WOE

        # Get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']

            if attribute == 'Missing':
                WOE_map_dict['Missing'][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict['Missing'][char] = np.nan

    # Validate data
    print('Number of key : ', len(WOE_map_dict.keys()))
    print(WOE_map_dict)

    return WOE_map_dict

# Function to replace the raw data in the train set with WOE values
def transform_woe(raw_data, WOE_dict, num_cols,WOE_map_dict):
    
    woe_data = raw_data.copy()

    # Map the raw data
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    # Map the raw data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])

    return woe_data

def forward(X, y, predictors, scoring='roc_auc', cv=5):
    """
    Function to perform forward selection procedure.

    Parameters
    ----------
    X : {array-like} of shape (n_sample, n_predictors)
      All predictors set.

    y : {array-like} of shape (n_sample, )
      The dependent or response variable.

    predictors : {array-like} of shape (n_sample, )
      Index of predictors

    scoring : a single {str}, default='roc_auc'
      The scoring parameter based on scikit-learn cross_validate documentation.

    cv : int, default=5
      Number of folds for k-Fold CV.

    Returns
    -------
    models : {array-like} of shape (n_combinations, k)
      Summary of predictors and its AIC score for each possible combination.

    best_model : {array-like} of shape (2, )
      Best model of models with the smallest AIC score.
    """

    # Initialize list of results
    results = []

    # Define sample size and  number of all predictors
    n_samples, n_predictors = X.shape

    # Define list of all predictors
    col_list = np.arange(n_predictors)

    # Define remaining predictors for each k
    remaining_predictors = [p for p in col_list if p not in predictors]

    # Initialize list of predictors and its CV Score
    pred_list = []
    score_list = []

    # Cross validate each possible combination of remaining predictors
    for p in remaining_predictors:
        combi = predictors + [p]

        # Extract predictors combination
        X_ = X[:, combi]
        y_ = y

        # Define the estimator
        model = LogisticRegression(penalty = None,
                                   class_weight = 'balanced')

        # Cross validate the recall scores of the model
        cv_results = cross_validate(estimator = model,
                                    X = X_,
                                    y = y_,
                                    scoring = scoring,
                                    cv = cv)

        # Calculate the average CV/recall score
        score_ = np.mean(cv_results['test_score'])

        # Append predictors combination and its CV Score to the list
        pred_list.append(list(combi))
        score_list.append(score_)

    # Tabulate the results
    models = pd.DataFrame({"Predictors": pred_list,
                           "Recall": score_list})

    # Choose the best model
    best_model = models.loc[models['Recall'].argmax()]

    return models, best_model

def create_train_null(X_train,y_train):
    # Define predictor for the null model
    predictor = []

    # The predictor in the null model is zero values for all predictors
    X_null = np.zeros((X_train.shape[0], 1))

    # Define the estimator
    model = LogisticRegression(penalty = None,
                           class_weight = 'balanced')

    # Cross validate
    cv_results = cross_validate(estimator = model,
                                X = X_null,
                                y = y_train,
                                cv = 10,
                                scoring = 'recall')

    # Calculate the average CV/recall score
    score_ = np.mean(cv_results['test_score'])

     #Create table for the best model of each k predictors
    # Append the results of null model
    forward_models = pd.DataFrame({"Predictors": [predictor],
                                   "Recall": [score_]})
    print(forward_models)
    return forward_models

def list_predictors(X_train,y_train,forward_models):
    # Define list of predictors
    predictors = []
    n_predictors = X_train.shape[1]

    # Perform forward selection procedure for k=1,...,11 predictors
    for k in range(n_predictors):
        _, best_model = forward(X = X_train,
                                y = y_train,
                                predictors = predictors,
                                scoring = 'recall',
                                cv = 10)

        # Tabulate the best model of each k predictors
        forward_models.loc[k+1] = best_model
        predictors = best_model['Predictors']
    print(forward_models)
    return forward_models, predictors

def create_best_model(X_train,y_train,best_predictors,raw_train,best_idx):
    # Define X with best predictors
    X_train_best = X_train[:, best_predictors]

    # Fit best model
    best_model = LogisticRegression(penalty = None,
                                class_weight = 'balanced')
    best_model.fit(X_train_best, y_train)

    best_model_intercept = pd.DataFrame({'Estimate': best_model.intercept_},
                                    index = ["Intercept"])
    print(best_model_intercept)
    best_model_params = raw_train.columns[best_predictors].tolist()
    best_model_coefs = pd.DataFrame({'Estimate':  np.reshape(best_model.coef_, best_idx)},
                                index = best_model_params)

    best_model_summary = pd.concat((best_model_intercept, best_model_coefs),
                               axis = 0)

    print(best_model_summary)
    return best_model, X_train_best

def create_best_model_summary(X_train,y_train,forward_models,predictors,raw_train):
    best_predictors = forward_models['Predictors'].loc[len(predictors)]
    # Define X with best predictors
    X_train_best = X_train[:, best_predictors]

    # Fit best model
    best_model = LogisticRegression(penalty = None,
                                    class_weight = 'balanced')
    best_model.fit(X_train_best, y_train)
    best_model_intercept = pd.DataFrame({'Characteristic': 'Intercept',
                                     'Estimate': best_model.intercept_})
    print(best_model_intercept)
    best_model_params = raw_train.columns[best_predictors].tolist()
    best_model_coefs = pd.DataFrame({'Characteristic':  best_model_params,
                                     'Estimate':        np.reshape(best_model.coef_,
                                                                   len(best_predictors))})

    best_model_summary = pd.concat((best_model_intercept, best_model_coefs),
                                   axis = 0,
                                   ignore_index = True)

    print(best_model_summary)
    return best_model_summary

def predict_proba_train(X_train_best,y_train,best_model):
    # Predict probability of default on X_train
    y_train_pred_proba = best_model.predict_proba(X_train_best)

    # Calculate sensitivity (TPR) and 1-specificity (FPR) from each possible threshold
    fpr, tpr, threshold = metrics.roc_curve(y_true = y_train,
                                            y_score = y_train_pred_proba[:,1])

    # Calculate AUC score using method sklearn.metrics.roc_auc_score
    auc_train = metrics.roc_auc_score(y_true = y_train,
                                      y_score = y_train_pred_proba[:,1])
    auc_train = round(auc_train, 2)

    # Plot ROC and its AUC
    plt.plot(fpr,
             tpr,
             label = "AUC train="+str(auc_train))

    plt.ylabel("Sensitivity/Recall")
    plt.xlabel("False Positive Rate (1- Specificity)")
    plt.legend(loc = 4)
    plt.show()
    skplt.metrics.plot_ks_statistic(y_train,
                                    y_train_pred_proba)
    plt.show()
    
    return y_train_pred_proba, auc_train

def y_test_pred_proba(X_test_best,y_test,best_model):
    # Predict probability of default on X_test
    y_test_pred_proba = best_model.predict_proba(X_test_best)

    # Calculate sensitivity (TPR) and 1-specificity (FPR) from each possible threshold
    fpr_, tpr_, threshold_ = metrics.roc_curve(y_true = y_test,
                                            y_score = y_test_pred_proba[:,1])

    # Calculate AUC score using method sklearn.metrics.roc_auc_score
    auc_test = metrics.roc_auc_score(y_true = y_test,
                                    y_score = y_test_pred_proba[:,1])
    auc_test = round(auc_test, 2)

    # Plot ROC and its AUC
    plt.plot(fpr_,
            tpr_,
            label = "AUC test="+str(auc_test))

    plt.ylabel("Sensitivity/Recall")
    plt.xlabel("False Positive Rate (1- Specificity)")
    plt.legend(loc = 4)
    plt.show()
    # Plot KS Statistic
    skplt.metrics.plot_ks_statistic(y_test,
                                    y_test_pred_proba)
    plt.show()
    return y_test_pred_proba, auc_test

def create_factor_offset():
    # Define Factor and Offset
    factor = 20/np.log(2)
    offset = 300-(factor*np.log(30))

    print(f"Offset = {offset:.2f}")
    print(f"Factor = {factor:.2f}")  
    return factor,offset

def create_scorecards(factor,offset,forward_models,predictors,best_model,best_model_summary,WOE_table):
    num_columns = ['person_age',
                   'person_income',
                   'person_emp_length',
                   'loan_amnt',
                   'loan_int_rate',
                   'loan_percent_income',
                   'cb_person_cred_hist_length']

    best_predictors = forward_models['Predictors'].loc[len(predictors)]

    # Define n = number of characteristics
    n = len(best_predictors)

    # Define b0
    b0 = best_model.intercept_[0]

    print(f"n = {n}")
    print(f"b0 = {b0:.4f}")

    # Adjust characteristic name in best_model_summary_table
    for col in best_model_summary['Characteristic']:
        if col in num_columns:
            bin_col = col + '_bin'
        else:
            bin_col = col
        best_model_summary.replace(col, bin_col, inplace = True)


        # Merge tables to get beta_i for each characteristic
        scorecards = pd.merge(left = WOE_table,
                            right = best_model_summary,
                            how = 'left',
                            on = ['Characteristic'])

    print(scorecards.head())

    # Define beta and WOE
    beta = scorecards['Estimate']
    WOE = scorecards['WOE']

    # Calculate the score point for each attribute
    scorecards['Points'] = (offset/n) - factor*((b0/n) + (beta*WOE))
    scorecards['Points'] = scorecards['Points'].astype('int')

    print(scorecards)

    return scorecards



def main():

    X_train,X_test,y_train,y_test,response_variable=loaddata('/Users/rianrachmanto/pypro/project/credit-scoring-analysis/data/credit_risk_dataset.csv')
    data_train_binned=databinning(X_train, y_train,response_variable)
    crosstab_list=create_crosstablist(data_train_binned,'loan_status') # replace response_variable with 'loan_status'
    IV_list,WOE_list=create_woe_iv(crosstab_list)
    WOE_table,IV_table=create_woe_table(crosstab_list,IV_list)
    WOE_map_dict=get_woe_map_dict(WOE_table)
    num_columns = ['person_age',
                   'person_income',
                   'person_emp_length',
                   'loan_amnt',
                   'loan_int_rate',
                   'loan_percent_income',
                   'cb_person_cred_hist_length']
    woe_train = transform_woe(raw_data = X_train,
                              WOE_dict = WOE_map_dict,
                              num_cols = num_columns,
                              WOE_map_dict=WOE_map_dict)
    print(woe_train)
    woe_test = transform_woe(raw_data = X_test,
                             WOE_dict = WOE_map_dict,
                             num_cols = num_columns,
                             WOE_map_dict=WOE_map_dict)
    print(woe_test)

    raw_train=X_train
    print(raw_train)

    X_train=woe_train.to_numpy()
    print(X_train)

    y_train=y_train.to_numpy()
    print(y_train)

    forward_models=create_train_null(X_train,y_train)
    forward_models,predictors=list_predictors(X_train,y_train,forward_models)
    
    # Find the best Recall score
    best_idx = forward_models['Recall'].argmax()
    best_recall = forward_models['Recall'].loc[best_idx]
    best_predictors = forward_models['Predictors'].loc[best_idx]

    # Print the summary
    print('Best index            :', best_idx)
    print('Best Recall           :', best_recall)
    print('Best predictors (idx) :', best_predictors)
    print('Best predictors       :')
    print(raw_train.columns[best_predictors].tolist())

    best_model, X_train_best=create_best_model(X_train,y_train,best_predictors,raw_train,best_idx)

    # Predict class labels for sample in X_train.
    y_train_pred = best_model.predict(X_train_best)
    print(y_train_pred)

    # Calculate the recall score on the train set
    recall_train = recall_score(y_true = y_train,
                                y_pred = y_train_pred)
    
    y_train_pred_proba = best_model.predict_proba(X_train_best)[:,[1]]
    print(y_train_pred_proba)

    print(recall_train)

    raw_test=X_test
    X_test=woe_test.to_numpy()
    y_test=y_test.to_numpy()

    # Define X_test with best predictors
    X_test_best = X_test[:, best_predictors]

    # Predict class labels for sample in X_test.
    y_test_pred = best_model.predict(X_test_best)
    print(y_test_pred)

    # Calculate the recall score on the test set
    recall_test = recall_score(y_true = y_test,
                           y_pred = y_test_pred)

    print(recall_test)

    # Predict the probability estimates
    y_test_pred_proba = best_model.predict_proba(X_test_best)[:,[1]]
    print(y_test_pred_proba)

    best_model_summary=create_best_model_summary(X_train,y_train,forward_models,predictors,raw_train)

    # Predict class labels for sample in X_train.
    y_train_pred = best_model.predict(X_train_best)
    print(y_train_pred)

    # Calculate the recall score on the train set
    recall_train = recall_score(y_true = y_train,
                                y_pred = y_train_pred)

    print(recall_train)

    y_train_pred_proba, auc_train=predict_proba_train(X_train_best,y_train,best_model)

    # Define X_test with best predictors
    X_test_best = X_test[:, best_predictors]

    # Predict class labels for sample in X_test.
    y_test_pred = best_model.predict(X_test_best)
    print(y_test_pred)

    
    y_test_pred_proba = best_model.predict_proba(X_test_best)

    # Calculate sensitivity (TPR) and 1-specificity (FPR) from each possible threshold
    fpr_, tpr_, threshold_ = metrics.roc_curve(y_true = y_test,
                                            y_score = y_test_pred_proba[:,1])

    # Calculate AUC score using method sklearn.metrics.roc_auc_score
    auc_test = metrics.roc_auc_score(y_true = y_test,
                                    y_score = y_test_pred_proba[:,1])
    auc_test = round(auc_test, 2)

    # Plot ROC and its AUC
    plt.plot(fpr_,
            tpr_,
            label = "AUC test="+str(auc_test))

    plt.ylabel("Sensitivity/Recall")
    plt.xlabel("False Positive Rate (1- Specificity)")
    plt.legend(loc = 4)
    plt.show()
    # Plot KS Statistic
    skplt.metrics.plot_ks_statistic(y_test,
                                    y_test_pred_proba)
    plt.show()

    factor,offset=create_factor_offset()
    scorecards=create_scorecards(factor,offset,forward_models,predictors,best_model,best_model_summary,WOE_table)
    #save scorecards as pickle
    scorecards.to_pickle('/Users/rianrachmanto/pypro/project/credit-scoring-analysis/model/scorecards.pkl')

    grouped_char = scorecards.groupby('Characteristic')
    grouped_points = grouped_char['Points'].agg(['min', 'max'])
    print(grouped_points)

    # Calculate the min and max score from the scorecards
    total_points = grouped_points.sum()
    min_score = total_points['min']
    max_score = total_points['max']

    print(f"The lowest credit score = {min_score}")
    print(f"The highest credit score = {max_score}")

        
if __name__ == "__main__":
    main()


   
