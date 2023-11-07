# Logistic Regression in Building Credit Scorecard
## Problem Statement
In order to reduce the number of NPL, it is necessary to measure creditworthiness of a borrower using credit score. This project is aiming at giving a mean to lender for measuring credit score using scorecard.
## Project Objective
Build credit scorecard using machine learning (Logistic Regression).
## Project Workflow
This project has 4 main steps:
### 1. Binning
This step is to create binning for numerical data (for instance age or income)
### 2. Calculate WoE and IV
Weight of Evidence (WoE) is measurement of predictive powers of predictors, while Information Values estimates the strength of predictors.
### 3. Forward Selection
This steps is to find the best predictors and models using forward selection method. This step involves Logistic Regression in determining best preditors
### 4. Scaling
Scaling is the final step where the best models and predictores are given the score. The method using pdo. The end result is the scorecard.

This project has been writen as article in medium:
https://medium.com/@rachmanto.rian/logistic-regression-in-building-credit-scorecard-924bece9f953

and you can access the application (using streamlit) here:
https://creditscorecard.streamlit.app/

