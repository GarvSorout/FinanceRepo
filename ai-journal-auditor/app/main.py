import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="AI Journal Auditor", layout="wide")


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(
        "/Users/garvsorout/FinanceRepo/ai-journal-auditor/model/audit_model.pkl"
    )


pipeline = load_model()


st.title("📊 AI-Powered Journal Entry Auditor")

uploaded_file = st.file_uploader("Upload your journal entry CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Feature engineering: add 'Hour'
    df["Hour"] = pd.to_datetime(df["Timestamp"]).dt.hour

    # Select same features as during training
    features = df[["Account", "Debit/Credit", "Amount", "Preparer", "Approver", "Hour"]]

    # Run model
    preds = pipeline.predict(features)
    df["IsAnomaly"] = preds == -1

    st.success(f"✅ Analysis complete! {df['IsAnomaly'].sum()} anomalies flagged.")

    # Show flagged entries
    flagged = df[df["IsAnomaly"] == True]
    st.dataframe(flagged, use_container_width=True)

    # Download results
    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download flagged anomalies", csv, "flagged_anomalies.csv", "text/csv"
    )
