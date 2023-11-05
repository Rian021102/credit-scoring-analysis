import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
def loaddata(path):
    df=pd.read_csv(path)
    print(df.T)
    response_variable=df['loan_status']
    y=response_variable
    X=df.drop(columns=['loan_status'],axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

def eda_train(X_train, y_train):
    num_columns = ['person_age',
                   'person_income',
                   'person_emp_length',
                   'loan_amnt',
                   'loan_int_rate',
                   'loan_percent_income',
                   'cb_person_cred_hist_length']

    data_train = pd.concat([X_train, y_train], axis=1)

    for col in num_columns:
        # Print column name
        print(f"Column: {col}")

        # Print descriptive statistics by loan_status
        print(data_train.groupby('loan_status')[col].describe())
        print('------------------------------------------------------')

        # Plot histogram with hue by loan_status
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data_train, x=col, hue='loan_status', bins=20, kde=False, element="step", common_norm=False)
        plt.title(f"{col} Histogram with Hue by loan_status")
        plt.show()

        # Plot boxplot with hue by loan_status
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data_train['loan_status'], y=data_train[col])
        plt.title(f"{col} Boxplot with Hue by loan_status")
        plt.show()

        # Plot KDE plot with hue by loan_status
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data_train[col][data_train['loan_status'] == 0], shade=True, label='loan_status=0')
        sns.kdeplot(data_train[col][data_train['loan_status'] == 1], shade=True, label='loan_status=1')
        plt.title(f"{col} KDE Plot with Hue by loan_status")
        plt.show()

    # Plot correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data_train.corr(), annot=True)
    plt.show()

    return data_train

def main():
    X_train,X_test,y_train,y_test=loaddata('/Users/rianrachmanto/pypro/project/credit-scoring-analysis/data/credit_risk_dataset.csv')
    data_train=eda_train(X_train,y_train)

if __name__=='__main__':
    main()
