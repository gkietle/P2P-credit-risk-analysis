# Create streamlit app for the project Loan Prediction Lending Club

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def calculate_installment(loan_amnt, int_rate, term):
    r = int_rate / 100 / 12
    n = int(term.split()[0])
    installment = loan_amnt * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    st.write(f'The monthly installment is: {installment}')
    return installment


def calculate_dti(annual_inc, installment):
    dti = (installment * 12) / annual_inc
    st.write(f'The debt-to-income ratio is: {dti}')
    return dti


# Load the model (assuming you have multiple models in 'Models' directory)
models = {
    "logistic_regression": joblib.load(open('../models checkpoint/log_reg_baseline.pkl', 'rb')),
    # "Naive Bayes": pickle.load(open('./Models/nb_baseline.pkl', 'rb')),
    "SVM": joblib.load(open('../models checkpoint/svc_baseline.pkl', 'rb')),
    "Random Forest": joblib.load(open('../models checkpoint/rf_baseline.pkl', 'rb')),
    "Decision Tree": joblib.load(open('../models checkpoint/dt_baseline.pkl', 'rb')),
    "XGBoost": joblib.load(open('../models checkpoint/xgb_baseline.pkl', 'rb'))
}


# Create input form
def main():
    st.title('Loan Prediction Lending Club')
    st.write('This is a simple loan prediction app that predicts whether a loan will be paid off or not based on a few features.')
    st.write('Please fill in the form below to get the prediction.')

    # Side bar for model selection
    selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))
    model = models[selected_model]

    # Read df
    df = pd.read_csv('./dataset/lending_club_loan_two.csv')
    df = df.drop('loan_status', axis=1)
    df = df.dropna()

    with st.form(key='columns_in_form'):
        c1, c2, c3 = st.columns(spec=[5, 5, 4], gap="small")
        with c1:
            # Loan characteristics
            st.header('Loan 💸')
            loan_amnt = st.number_input('Loan Amount', min_value=df['loan_amnt'].min(
            ), max_value=df['loan_amnt'].max(), step=100.0)
            term = st.selectbox('Term', ['36 months', '60 months'])
            int_rate = st.slider('Interest Rate', min_value=df['int_rate'].min(
            ), max_value=df['int_rate'].max(), step=0.01, format='%f')
            grade = st.selectbox('Grade', df['grade'].unique())
            sub_grade = st.slider(
                'Sub Grade', min_value=1, max_value=5, step=1)
            purpose = st.selectbox('Purpose', df['purpose'].unique())
            application_type = st.selectbox(
                'Application Type', df['application_type'].unique())
        with c2:
            # Borrower characteristics
            st.header('Borrower 🫂')
            emp_title = st.text_input('Employment Title')
            emp_length = st.selectbox(
                'Employment Length', df['emp_length'].unique())
            home_ownership = st.selectbox(
                'Home Ownership', ['MORTGAGE', 'RENT', 'OWN', 'OTHER'])
            annual_inc = st.number_input('Annual Income', min_value=df['annual_inc'].min(
            ), max_value=df['annual_inc'].max(), step=1000.0)
            verification_status = st.selectbox(
                'Verification Status', df['verification_status'].unique())
            address = st.text_input('Address')
            zip_code = st.selectbox(
                'Zip Code', df['address'].apply(lambda x: x[-5:]).unique())
        with c3:
            # Credit characteristics
            st.header('Credit 📊')
            df['earliest_cr_line_year'] = pd.to_datetime(
                df['earliest_cr_line']).dt.year
            earliest_cr_line = st.slider('Year of Earliest Credit Line', min_value=df['earliest_cr_line_year'].min(
            ), max_value=df['earliest_cr_line_year'].max(), step=1)
            open_acc = st.number_input('Open Accounts', min_value=df['open_acc'].min(
            ), max_value=df['open_acc'].max(), step=1.0)
            pub_rec = st.radio('Public Records', ["No", "Yes"], index=0)
            revol_bal = st.slider('Revolving Balance', min_value=df['revol_bal'].min(
            ), max_value=df['revol_bal'].max(), step=100.0)
            revol_util = st.slider('Revolving Utilization', min_value=df['revol_util'].min(
            ), max_value=df['revol_util'].max(), step=0.01)
            total_acc = st.number_input('Total Accounts', min_value=df['total_acc'].min(
            ), max_value=df['total_acc'].max(), step=1.0)
            initial_list_status = st.selectbox(
                'Initial List Status', df['initial_list_status'].unique())

            mort_acc = st.radio('Mortgage Accounts', ["No", "Yes"], index=0)
            pub_rec_bankruptcies = st.radio(
                'Public Records Bankruptcies', ["No", "Yes"], index=0)

        submitButton = st.form_submit_button(
            label='Calculate & Make Prediction')
    if submitButton:
        # Create a dictionary to hold the data
        data = {
            'loan_amnt': loan_amnt,
            'term': term,
            'int_rate': int_rate,
            'grade': grade,
            'sub_grade': sub_grade,
            'emp_title': emp_title,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'annual_inc': annual_inc,
            'verification_status': verification_status,
            'purpose': purpose,
            'earliest_cr_year': earliest_cr_line,
            'open_acc': open_acc,
            'pub_rec': pub_rec,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'total_acc': total_acc,
            'initial_list_status': initial_list_status,
            'application_type': application_type,
            'mort_acc': mort_acc,
            'pub_rec_bankruptcies': pub_rec_bankruptcies,
            'address': address,
            'zip_code': zip_code
        }

        X = pd.DataFrame(data, index=[0])
        X.drop('emp_title', axis=1, inplace=True)
        X.drop('emp_length', axis=1, inplace=True)
        X.drop('address', axis=1, inplace=True)
        X['term'] = X['term'].map({'36 months': 36, '60 months': 60})
        # Concat grade and sub_grade
        X['sub_grade'] = X['grade'] + X['sub_grade'].astype(str)
        X['sub_grade'] = X['sub_grade'].map({'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5, 'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10, 'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15, 'D1': 16,
                                            'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20, 'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25, 'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30, 'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35})
        X.drop('grade', axis=1, inplace=True)
        X['pub_rec'] = X['pub_rec'].map({'No': 0, 'Yes': 1})
        X['mort_acc'] = X['mort_acc'].map({'No': 0, 'Yes': 1})
        X['pub_rec_bankruptcies'] = X['pub_rec_bankruptcies'].map(
            {'No': 0, 'Yes': 1})

        # Calculate the monthly installment
        X['installment'] = calculate_installment(loan_amnt, int_rate, term)
        X['dti'] = calculate_dti(annual_inc, X['installment'].values[0])

        # Create dummy variables base on the categorical variables
        df_extend = pd.DataFrame(columns=['verification_status_Source Verified', 'verification_status_Verified', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
                                 'purpose_vacation', 'purpose_wedding', 'initial_list_status_w', 'application_type_INDIVIDUAL', 'application_type_JOINT', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'zip_code_05113', 'zip_code_11650', 'zip_code_22690', 'zip_code_29597', 'zip_code_30723', 'zip_code_48052', 'zip_code_70466', 'zip_code_86630', 'zip_code_93700'], index=[0])

        df_extend = df_extend.fillna(0)
        dummies_columns = ['verification_status', 'purpose',
                           'initial_list_status', 'application_type', 'home_ownership', 'zip_code']

        for col in dummies_columns:
            index = col + '_' + X[col].iloc[0]
            if index in df_extend.columns:
                df_extend[index] = 1
            print(index)

        X = pd.concat([X, df_extend], axis=1)
        X.drop(columns=dummies_columns, inplace=True)

        df_temp = pd.read_csv('./dataset/lending_club_loan_cleaned.csv')
        Input_features = df_temp.drop('loan_status', axis=1).columns.values
        X_temp = df_temp[Input_features]
        X = X.reindex(columns=X_temp.columns, fill_value=0)

        scaler = MinMaxScaler()
        scaler.fit(X_temp[Input_features])
        X_temp[Input_features] = scaler.transform(X_temp)

        X = scaler.transform(X)

        # Make predictions
        prediction = model.predict(X)

        # Display the prediction
        if prediction == 0:
            st.error('This loan will be Charged off', icon="🚨")
        else:
            st.success('This loan will be Fully Paid', icon="✅")


if __name__ == '__main__':
    main()
