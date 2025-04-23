import streamlit as st
import joblib
import pandas as pd

# Load the model and label encoders
model = joblib.load('rf_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')
expected_columns = ['job', 'marital', 'education', 'default', 'housing',
                    'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Check if all expected encoders are present
missing_columns = set(expected_columns) - set(label_encoders.keys())
if missing_columns:
    st.write("Missing columns in label encoders:", missing_columns)
else:
    st.write("All required columns are present in label encoders.")

# Title
st.title('Bank Marketing Prediction')

# Collecting user input
def get_user_input():
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    job = st.selectbox('Job', options=list(label_encoders['job'].classes_))
    marital = st.selectbox('Marital', options=list(label_encoders['marital'].classes_))
    education = st.selectbox('Education', options=list(label_encoders['education'].classes_))
    default = st.selectbox('Default', options=list(label_encoders['default'].classes_))
    housing = st.selectbox('Housing', options=list(label_encoders['housing'].classes_))
    loan = st.selectbox('Loan', options=list(label_encoders['loan'].classes_))
    contact = st.selectbox('Contact', options=list(label_encoders['contact'].classes_))
    month = st.selectbox('Month', options=list(label_encoders['month'].classes_))
    day_of_week = st.selectbox('Day of Week', options=list(label_encoders['day_of_week'].classes_))
    duration = st.number_input('Duration', min_value=0, step=10, value=1)
    campaign = st.number_input('Campaign', min_value=1, step=1, value=1)
    pdays = st.number_input('Pdays', min_value=0, step=1, value=999)
    previous = st.number_input('Previous', min_value=0, step=1, value=0)
    poutcome = st.selectbox('Poutcome', options=list(label_encoders['poutcome'].classes_))
    emp_var_rate = st.number_input('Employment Variation Rate', value=1.0)
    cons_price_idx = st.number_input('Consumer Price Index', value=93.994)
    cons_conf_idx = st.number_input('Consumer Confidence Index', value=-36.4)
    euribor3m = st.number_input('Euribor 3 Month Rate', value=4.857)
    nr_employed = st.number_input('Number of Employees', value=5191.0)

    user_input = {
        'age': age,
        'job': label_encoders['job'].transform([job])[0],
        'marital': label_encoders['marital'].transform([marital])[0],
        'education': label_encoders['education'].transform([education])[0],
        'default': label_encoders['default'].transform([default])[0],
        'housing': label_encoders['housing'].transform([housing])[0],
        'loan': label_encoders['loan'].transform([loan])[0],
        'contact': label_encoders['contact'].transform([contact])[0],
        'month': label_encoders['month'].transform([month])[0],
        'day_of_week': label_encoders['day_of_week'].transform([day_of_week])[0],
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': label_encoders['poutcome'].transform([poutcome])[0],
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
    }
    
    return pd.DataFrame([user_input])

# Get user input
user_input_df = get_user_input()

# Prediction
if st.button('Predict'):
    prediction = model.predict(user_input_df)
    prediction_proba = model.predict_proba(user_input_df)

    predicted_label = label_encoders['y'].inverse_transform(prediction)[0]
    proba_yes = prediction_proba[0][1]
    st.write(f"Prediction: {predicted_label}")
    st.write(f"Probability of Yes: {proba_yes:.2f}")
