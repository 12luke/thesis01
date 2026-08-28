import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

# Set Page Config
st.set_page_config(
    page_title="Financial Anomaly & Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Financial Anomaly & Fraud Detection Evaluation Framework")
st.markdown("Upload accounting or transaction datasets (CSV or Excel) to evaluate PyOD unsupervised models with zero index or row loss.")

# ---------------------------------------------------------
# 1. Feature Engineering (Index-Preserving & NaNs Safe)
# ---------------------------------------------------------
def engineer_features(df):
    """Engineers behavioral risk features while preserving 100% of input rows."""
    data = df.copy()
    
    # Case-insensitive column lookup
    cols_lower = {str(c).lower().strip(): c for c in data.columns}
    
    amount_col = next((cols_lower[c] for c in cols_lower if 'amount' in c), None)
    old_orig_col = next((cols_lower[c] for c in cols_lower if 'oldbalance' in c and ('org' in c or 'orig' in c or 'src' in c)), None)
    if not old_orig_col:
        old_orig_col = next((cols_lower[c] for c in cols_lower if 'oldbalance' in c), None)
        
    new_orig_col = next((cols_lower[c] for c in cols_lower if 'newbalance' in c and ('org' in c or 'orig' in c or 'src' in c)), None)
    if not new_orig_col:
        new_orig_col = next((cols_lower[c] for c in cols_lower if 'newbalance' in c), None)
        
    old_dest_col = next((cols_lower[c] for c in cols_lower if 'oldbalance' in c and 'dest' in c), None)
    new_dest_col = next((cols_lower[c] for c in cols_lower if 'newbalance' in c and 'dest' in c), None)
    step_col = next((cols_lower[c] for c in cols_lower if 'step' in c or 'time' in c or 'date' in c), None)
    name_orig_col = next((cols_lower[c] for c in cols_lower if 'nameorig' in c or 'origin' in c or 'account' in c), None)

    # Coerce core numeric columns and fill NaNs immediately
    core_cols = [c for c in [amount_col, old_orig_col, new_orig_col, old_dest_col, new_dest_col, step_col] if c is not None]
    for c in core_cols:
        data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0.0)

    if amount_col and old_orig_col and new_orig_col:
        # 1. Balance Delta Error (BDE)
        data['BDE'] = data[old_orig_col] - data[new_orig_col] - data[amount_col]
        
        # 2. Transaction Velocity (TV) — Safe index preservation
        if name_orig_col:
            data['TV'] = data.groupby(name_orig_col)[amount_col].transform('count')
        else:
            data['TV'] = 1.0
            
        # 3. Amount-to-Balance Ratio (ABR)
        data['ABR'] = data[amount_col] / (data[old_orig_col] + 1.0)
        
        # 4. Round Number Indicator (RNI)
        data['RNI'] = (data[amount_col] % 1000 == 0).astype(int)
        
        # 5. Additional Domain Features
        data['LogAmount'] = np.log1p(np.maximum(0, data[amount_col]))
        
        if old_dest_col and new_dest_col:
            data['DestBalanceDiff'] = data[new_dest_col] - data[old_dest_col]
        else:
            data['DestBalanceDiff'] = 0.0
            
        if step_col:
            data['Hour'] = data[step_col] % 24
        else:
            data['Hour'] = 0.0

        # One-Hot Encoding for 'type'
        type_col = next((cols_lower[c] for c in cols_lower if 'type' in c and c != 'timestamp'), None)
        if type_col:
            data = pd.get_dummies(data, columns=[type_col], drop_first=True, dtype=int)

        features = [
            amount_col, old_orig_col, new_orig_col,
            'BDE', 'TV', 'ABR', 'RNI', 'LogAmount', 'DestBalanceDiff', 'Hour'
        ]
        if old_dest_col: features.append(old_dest_col)
        if new_dest_col: features.append(new_dest_col)
        features += [col for col in data.columns if col.startswith('type_')]
    else:
        features = []

    # Final cleanup ensuring zero NaN values across selected features
    for col in features:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)

    if not features:
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)
        features = data.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [f for f in features if f in data.columns and np.issubdtype(data[f].dtype, np.number)]

    return data, features

# ---------------------------------------------------------
# 2. File Upload & Robust Encoding / Format Reader
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    raw_df = None

    # Handle Excel Formats
    if filename.endswith(('.xlsx', '.xls')):
        try:
            raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
    else:
        # Handle CSV with Fallback Encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        for enc in encodings:
            try:
                uploaded_file.seek(0)  # Reset file buffer position before re-reading
                raw_df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        # Fallback: Check if file was actually an Excel file renamed as CSV
        if raw_df is None:
            try:
                uploaded_file.seek(0)
                raw_df = pd.read_excel(uploaded_file)
            except Exception:
                st.error("Could not parse file. Please verify that the file format matches its extension and uses standard character encoding.")

    if raw_df is not None:
        st.success(f"Successfully loaded dataset with **{len(raw_df)}** rows and **{raw_df.shape[1]}** columns.")

        with st.spinner("Engineering features and aligning indices..."):
            proc_df, feature_cols = engineer_features(raw_df)

        # Detect Label Column
        label_candidates = [c for c in raw_df.columns if str(c).lower().strip() in ['isfraud', 'label', 'class', 'anomaly', 'target']]
        label_col = label_candidates[0] if label_candidates else None

        # Sidebar Parameters
        st.sidebar.header("Model Configuration")
        contamination = st.sidebar.slider("Contamination Threshold", 0.001, 0.20, 0.05, step=0.005)
        selected_models = st.sidebar.multiselect(
            "Select Algorithms",
            ["Isolation Forest", "ECOD", "COPOD"],
            default=["Isolation Forest", "ECOD", "COPOD"]
        )

        if st.button("Run Anomaly Detection") and selected_models:
            X = proc_df[feature_cols].values
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            # Prepare True Labels safely
            if label_col:
                y_true = pd.to_numeric(raw_df[label_col], errors='coerce').fillna(0).astype(int).values
            else:
                y_true = None

            results = []
            preds_dict = {}

            for model_name in selected_models:
                if model_name == "Isolation Forest":
                    model = IForest(contamination=contamination, random_state=42)
                elif model_name == "ECOD":
                    model = ECOD(contamination=contamination)
                elif model_name == "COPOD":
                    model = COPOD(contamination=contamination)

                model.fit(X_scaled)
                preds = model.labels_
                preds_dict[model_name] = preds

                if y_true is not None:
                    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
                    results.append({
                        "Model": model_name,
                        "Evaluated Rows": len(preds),
                        "Precision": f"{p:.4f}",
                        "Recall": f"{r:.4f}",
                        "F1 Score": f"{f1:.4f}"
                    })

            # Summary Display
            st.subheader("Evaluation Results")
            if y_true is not None:
                st.table(pd.DataFrame(results))
                st.info(f"Verified: All **{len(raw_df)}** rows were successfully evaluated.")
            else:
                st.warning("No ground-truth label column (`isFraud`) detected. Displaying row evaluation counts only.")
                st.write(f"Total Evaluated Rows: **{len(X_scaled)}**")

            # Visualizations
            st.subheader("Model Predictions Breakdown")
            cols = st.columns(len(selected_models))
            for idx, (m_name, m_preds) in enumerate(preds_dict.items()):
                with cols[idx]:
                    st.markdown(f"**{m_name}**")
                    counts = pd.Series(m_preds).value_counts().rename({0: 'Normal', 1: 'Fraud'})
                    st.bar_chart(counts)
else:
    st.info("Please upload a CSV or Excel dataset to begin processing.")
