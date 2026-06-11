import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(BASE_DIR / 'models' / 'preprocessor.pkl')
    model        = joblib.load(BASE_DIR / 'models' / 'best_model.pkl')
    return preprocessor, model

preprocessor, model = load_artifacts()


st.title("Churn Predictor | Telecomunicações")
st.markdown("Insira os dados do cliente para prever a probabilidade de cancelamento.")

st.sidebar.header("Dados do cliente")

tenure          = st.sidebar.slider("Tempo de contrato (meses)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Cobrança mensal (R$)", 18.0, 120.0, 65.0)
total_charges   = st.sidebar.number_input("Cobrança total (R$)", 0.0, 9000.0,
                                           value=float(tenure * monthly_charges))
contract        = st.sidebar.selectbox("Tipo de contrato",
                                        ["Month-to-month", "One year", "Two year"])
internet        = st.sidebar.selectbox("Serviço de internet",
                                        ["Fiber optic", "DSL", "No"])
payment         = st.sidebar.selectbox("Método de pagamento",
                                        ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)",
                                         "Credit card (automatic)"])
tech_support    = st.sidebar.selectbox("Suporte técnico", ["No", "Yes"])
online_security = st.sidebar.selectbox("Segurança online",  ["No", "Yes"])
num_services    = st.sidebar.slider("Nº de serviços adicionais", 0, 6, 2)
senior          = st.sidebar.radio("Cliente sênior", ["No", "Yes"])
partner         = st.sidebar.radio("Tem parceiro(a)", ["No", "Yes"])
dependents      = st.sidebar.radio("Tem dependentes",  ["No", "Yes"])


input_data = pd.DataFrame([{
    'tenure':           tenure,
    'MonthlyCharges':   monthly_charges,
    'TotalCharges':     total_charges,
    'ChargePerTenure':  monthly_charges / (tenure + 1),
    'NumServices':      num_services,
    'gender':           'Male',
    'SeniorCitizen':    senior,
    'Partner':          partner,
    'Dependents':       dependents,
    'PhoneService':     'Yes',
    'MultipleLines':    'No',
    'PaperlessBilling': 'Yes',
    'IsNewCustomer':    1 if tenure <= 6 else 0,
    'OnlineSecurity':   online_security,
    'OnlineBackup':     'No',
    'DeviceProtection': 'No',
    'TechSupport':      tech_support,
    'StreamingTV':      'No',
    'StreamingMovies':  'No',
    'InternetService':  internet,
    'Contract':         contract,
    'PaymentMethod':    payment
}])

X_proc = preprocessor.transform(input_data)
prob   = model.predict_proba(X_proc)[0][1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Probabilidade de churn", f"{prob*100:.1f}%")
with col2:
    risk = "Alto" if prob > 0.6 else "Médio" if prob > 0.35 else "Baixo"
    color = "red" if prob > 0.6 else "orange" if prob > 0.35 else "green"
    st.markdown(f"**Nível de risco:** :{color}[{risk}]")
with col3:
    st.metric("Decisão do modelo", "Churn" if prob > 0.5 else "Retém")


    st.subheader("Por que essa previsão?")

explainer   = shap.LinearExplainer(model, X_proc,
                                    feature_perturbation='interventional')
shap_values = explainer.shap_values(X_proc)

feat_names = (
    ['tenure','MonthlyCharges','TotalCharges','ChargePerTenure','NumServices',
     'gender','SeniorCitizen','Partner','Dependents','PhoneService',
     'MultipleLines','PaperlessBilling','IsNewCustomer','OnlineSecurity',
     'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    + preprocessor.named_transformers_['cat']
       .named_steps['onehot'].get_feature_names_out(
           ['InternetService','Contract','PaymentMethod']).tolist()
)

shap_exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_proc[0],
    feature_names=feat_names
)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_exp, max_display=10, show=False)
st.pyplot(fig, bbox_inches='tight')
plt.close()