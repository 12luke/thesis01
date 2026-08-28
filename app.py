import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import HistGradientBoostingRegressor # Proxy/Placeholder for HBOS or custom ensemble

# Page Configuration
st.set_page_config(page_title="Audit ML Anomaly Diagnostic System", layout="wide")

st.title("🛡️ Continuous Auditing ML Anomaly Diagnostic System")
st.markdown("Upload transactional accounting data to evaluate model performance, inspect anomaly scores, and analyze high-risk exceptions.")

# --- SIDEBAR: Controls & Data Upload ---
st.sidebar.header("1. Upload Accounting Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File (e.g., PaySim or Ledger Data)", type=["csv"])

st.sidebar.header("2. Model Parameters")
contamination = st.sidebar.slider("Expected Anomaly Ratio (Contamination)", 0.001, 0.05, 0.01, step=0.001)

# --- FEATURE ENGINEERING PIPELINE ---
def engineer_features(df):
    """Engineers core behavioral risk features on raw transaction datasets."""
    data = df.copy()
    
    # Ensure standard column mapping fallback
    amount_col = [c for c in data.columns if 'amount' in c.lower()][0] if any('amount' in c.lower() for c in data.columns) else None
    old_orig_col = [c for c in data.columns if 'oldbalanceorg' in c.lower() or 'oldbalance' in c.lower()][0] if any('oldbalance' in c.lower() for c in data.columns) else None
    new_orig_col = [c for c in data.columns if 'newbalanceorig' in c.lower() or 'newbalance' in c.lower()][0] if any('newbalance' in c.lower() for c in data.columns) else None
    
    if amount_col and old_orig_col and new_orig_col:
        # Balance Delta Error (BDE)
        data['Balance_Delta_Error'] = np.abs((data[old_orig_col] - data[amount_col]) - data[new_orig_col])
        # Amount-to-Balance Ratio (ABR)
        data['ABR'] = data[amount_col] / (data[old_orig_col] + 1)
        # Round Number Indicator (RNI)
        data['RNI'] = (data[amount_col] % 100 == 0).astype(int)
        
        feature_cols = ['Balance_Delta_Error', 'ABR', 'RNI', amount_col]
        return data, feature_cols
    else:
        # Fallback to pure numeric columns if domain columns aren't matched exactly
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return data, numeric_cols

# --- MAIN WORKFLOW ---
if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded dataset: **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")
    
    # Preview Data
    with st.expander("🔍 Preview Uploaded Raw Data"):
        st.dataframe(df.head(10))
        
    # Process Features
    processed_df, feature_cols = engineer_features(df)
    
    st.subheader("⚙️ Feature Engineering & Model Execution")
    st.write(f"Engineered and selected features for isolation scoring: `{feature_cols}`")
    
    # Model Training & Inference Execution
    with st.spinner("Executing Unsupervised Anomaly Isolation Pipeline..."):
        X = processed_df[feature_cols].fillna(0)
        
        # Isolation Forest Execution
        iso_model = IsolationForest(contamination=contamination, random_state=42)
        processed_df['Anomaly_Raw_Score'] = -iso_model.fit_predict(X) # 1 for anomaly, -1 for normal converted to score
        processed_df['Anomaly_Flag'] = iso_model.predict(X)
        processed_df['Anomaly_Flag'] = processed_df['Anomaly_Flag'].map({1: 0, -1: 1}) # 1 = Anomaly
        
    # --- METRICS & REPORTING DASHBOARD ---
    total_records = len(processed_df)
    total_anomalies = processed_df['Anomaly_Flag'].sum()
    anomaly_pct = (total_anomalies / total_records) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records Evaluated", f"{total_records:,}")
    col2.metric("Flagged Exceptions", f"{total_anomalies:,}")
    col3.metric("Anomaly Ratio", f"{anomaly_pct:.2f}%")
    
    # Check if Target Labels Exist for Validation (e.g., 'isFraud')
    label_cols = [c for c in df.columns if c.lower() in ['isfraud', 'fraud', 'label', 'target']]
    if label_cols:
        target = label_cols[0]
        st.markdown("---")
        st.subheader("🎯 Ground-Truth Validation Metrics")
        
        from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
        
        y_true = processed_df[target]
        y_pred = processed_df['Anomaly_Flag']
        
        auc = roc_auc_score(y_true, processed_df['Anomaly_Raw_Score'])
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC Score", f"{auc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-Score", f"{f1:.3f}")
        
    # --- VISUALIZATIONS ---
    st.markdown("---")
    st.subheader("📊 Anomaly Score Distribution & Exception Analysis")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    
    # Distribution Plot
    sns.histplot(processed_df['Anomaly_Raw_Score'], kde=True, ax=ax[0], color="crimson")
    ax[0].set_title("Anomaly Score Distribution")
    ax[0].set_xlabel("Isolation Anomaly Score")
    
    # Exception Breakdown
    sns.countplot(x='Anomaly_Flag', data=processed_df, ax=ax[1], palette="coolwarm")
    ax[1].set_title("Normal (0) vs. Flagged Exception (1) Count")
    
    st.pyplot(fig)
    
    # --- HIGH-RISK EXCEPTION TABLE ---
    st.markdown("---")
    st.subheader("⚠️ Priority Audit Queue (Top Flagged Exceptions)")
    
    anomalies_df = processed_df[processed_df['Anomaly_Flag'] == 1].sort_values(
        by='Anomaly_Raw_Score', ascending=False
    )
    
    st.dataframe(anomalies_df)
    
    # Download Processed File
    csv_data = anomalies_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Priority Audit Exceptions as CSV",
        data=csv_data,
        file_name="audit_exceptions_priority_list.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a CSV dataset in the sidebar to run the continuous audit diagnostic pipeline.")
