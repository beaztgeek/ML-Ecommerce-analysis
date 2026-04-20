import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- CONFIGURATION & LOAD ---
st.set_page_config(page_title="E-commerce Intent Analytics", layout="wide")

@st.cache_resource
def load_assets():
    """Charge le modèle XGBoost et les métadonnées."""
    with open('sale_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model, model.feature_names_in_

model, features_model = load_assets()

# Références statistiques du dataset UCI (2017)
USER_BENCHMARKS = {
    'Returning_Visitor': {'conv': '13.9%', 'label': 'Returning', 'note': 'High volume, lower immediate intent.'},
    'New_Visitor': {'conv': '24.9%', 'label': 'Prospect', 'note': 'High intent, specific product search.'}
}

# --- SIDEBAR : PARAMETERS ---
st.sidebar.header("User Segmentation")

scenario = st.sidebar.selectbox("Predefined Scenarios", 
    ["Manual", "Paid Traffic Prospect", "Returning Customer", "High-Value Buyer"])

# Mapping des scénarios vers les variables d'entrée
config = {
    "Paid Traffic Prospect": (2.0, 0.020, 150, "New_Visitor", "Paid Ads", "May"),
    "Returning Customer": (0.0, 0.040, 300, "Returning_Visitor", "Direct", "May"),
    "High-Value Buyer": (60.0, 0.005, 150, "Returning_Visitor", "Direct", "Nov"),
    "Manual": (0.0, 0.100, 20, "Returning_Visitor", "Organic", "Jan")
}

pv, ex, dur, vis, src, mois_def = config[scenario]

# User behavior inputs
val_page = st.sidebar.slider("Page Value (Potential)", 0.0, 360.0, pv)
exit_rate = st.sidebar.slider("Exit Rate", 0.0, 0.2, ex, format="%.3f")
prod_duration = st.sidebar.number_input("Product Page Time (sec)", 0, 10000, dur)

st.sidebar.markdown("---")
st.sidebar.header("Contextual Data")
months = sorted([c.replace('Month_', '') for c in features_model if c.startswith('Month_')])
selected_month = st.sidebar.selectbox("Month", months, index=months.index(mois_def) if mois_def in months else 0)
visitor_type = st.sidebar.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor"], index=0 if vis == "Returning_Visitor" else 1)
traffic_source = st.sidebar.selectbox("Traffic Source", ["Organic", "Paid Ads", "Direct"])

# --- DATA PREPARATION ---
def build_feature_vector():
    df = pd.DataFrame(0, index=[0], columns=features_model)
    df['PageValues'] = val_page
    df['ExitRates'] = exit_rate
    df['ProductRelated_Duration'] = prod_duration
    df['Time_Per_Product_Page'] = prod_duration / 5 
    
    if f'Month_{selected_month}' in df.columns: df[f'Month_{selected_month}'] = 1
    if f'VisitorType_{visitor_type}' in df.columns: df[f'VisitorType_{visitor_type}'] = 1
    df['TrafficType'] = 2 if traffic_source == "Paid Ads" else 1
    return df

input_vector = build_feature_vector()

# --- MAIN DASHBOARD ---
st.title("E-commerce Purchase Intent Dashboard")
st.caption("Predictive analysis based on UCI Online Shoppers Dataset (Fiscal Year 2017)")

# Metrics row
k1, k2, k3 = st.columns(3)
k1.metric("Session Potential", f"{val_page:.1f} PV")
k2.metric("Acquisition Channel", traffic_source)
k3.metric("Seasonality", selected_month)

st.markdown("---")

# Prediction results
prob = float(model.predict_proba(input_vector)[0, 1])

col_left, col_right = st.columns([1, 2])

with col_left:
    st.write("### Model Output")
    st.metric("Conversion Probability", f"{prob:.1%}")
    st.progress(prob)

with col_right:
    st.write("### Strategic Recommendation")
    if prob > 0.70:
        st.success("**HIGH PRIORITY:** Lead is hot. Focus on frictionless checkout.")
    elif prob > 0.30:
        st.warning("**NURTURE:** Customer hesitating. Consider triggered discount or free shipping.")
    else:
        st.error("**LOW INTENT:** Top-of-funnel visitor. Focus on newsletter signups.")

# --- BUSINESS INSIGHTS ---
st.markdown("---")
st.write("### Behavioral Deep Dive")

c1, c2 = st.columns(2)

with c1:
    st.write("#### Returning Visitor Patterns")
    if visitor_type == "Returning_Visitor":
        st.info(f"""
        **Historical Context:** {USER_BENCHMARKS['Returning_Visitor']['note']}
        
        **Model Logic:** Returning users show higher 'window shopping' behavior. 
        Without a positive PageValue, the model remains conservative.
        """)
    else:
        st.write("*Inbound Prospect Analysis selected.*")

with c2:
    st.write("#### New Visitor Patterns")
    if visitor_type == "New_Visitor":
        st.success(f"""
        **Historical Context:** {USER_BENCHMARKS['New_Visitor']['note']}
        
        **Model Logic:** New users often arrive via direct product links. 
        Intent is typically higher and decision-making faster.
        """)
    else:
        st.write("*Loyalty Analysis selected.*")

# Reliability caption
if val_page == 0 and prob < 0.2:
    st.caption("Confidence: **High** (Statistical absence of basket value).")
elif val_page > 50 and prob > 0.8:
    st.caption("Confidence: **High** (Strong conversion pattern detected).")
else:
    st.caption("Confidence: **Medium** (Mixed behavioral signals).")

# --- DATASET FOOTER ---
with st.expander("Technical Metadata"):
    st.write("""
    - **Source:** UCI Machine Learning Repository.
    - **Sample:** 12,330 unique sessions.
    - **Key Driver:** PageValue represents the average value for a web page visited.
    """)