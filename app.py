import streamlit as st
import pandas as pd
import joblib

# 1. Configure the web page
st.set_page_config(page_title="HR Attrition Prediction", page_icon="🏢", layout="wide")
st.title("🏢 AI HR Attrition Prediction System")
st.markdown("---")

# 2. Load the trained AI models (Logistic Regression & Scaler)
@st.cache_resource
def load_models():
    model = joblib.load('hr_attrition_model.pkl')
    scaler = joblib.load('hr_scaler.pkl')
    return model, scaler

model, scaler = load_models()

# 3. Load and preprocess the raw data for querying
@st.cache_data
def load_data():
    df_raw = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df = df_raw.copy()

    # Apply the exact same cleaning steps done in Colab
    df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'])
    actual_attrition = df['Attrition'] # Hide the actual answers from the AI
    df = df.drop(columns=['Attrition'])

    # Data Encoding (Binary and One-Hot)
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_raw, df_encoded, actual_attrition

df_raw, df_encoded, actual_attrition = load_data()

# 4. Sidebar interface for Employee Selection
st.sidebar.header("🔍 Employee Lookup")
emp_list = df_raw['EmployeeNumber'].tolist()
selected_emp_id = st.sidebar.selectbox("Select Employee ID:", emp_list)

# Find the exact index of the selected employee
emp_index = df_raw[df_raw['EmployeeNumber'] == selected_emp_id].index[0]

# 5. Display Employee Profile
st.subheader(f"👤 Employee Profile: #{selected_emp_id}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Department", df_raw.loc[emp_index, 'Department'])
col2.metric("Job Role", df_raw.loc[emp_index, 'JobRole'])
col3.metric("Age", str(df_raw.loc[emp_index, 'Age']))
col4.metric("Monthly Income", f"${df_raw.loc[emp_index, 'MonthlyIncome']}")

# 6. AI Prediction Activation Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Analyze Risk with AI", type="primary"):
    
    # Extract the specific row for this employee and scale it
    emp_data = df_encoded.iloc[[emp_index]]
    emp_data_scaled = scaler.transform(emp_data)

    # Feed into AI for prediction
    prediction = model.predict(emp_data_scaled)[0]
    probability = model.predict_proba(emp_data_scaled)[0][1] * 100

    st.markdown("---")
    st.subheader("📊 Logistic Regression Model Results:")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: This employee is highly likely to LEAVE! (Probability: {probability:.1f}%)")
        st.info("💡 HR Recommendation: Schedule a 1-on-1 meeting immediately to discuss their concerns.")
    else:
        st.success(f"✅ SAFE: This employee shows good retention signs. (Leave Probability: {probability:.1f}%)")
        
    # Display the actual historical result for comparison
    actual_status = 'LEFT THE COMPANY' if actual_attrition[emp_index] == 'Yes' else 'STAYED'
    st.write(f"*Historical Actual Record: **{actual_status}***")