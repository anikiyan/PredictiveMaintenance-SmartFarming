# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Paths
DATA_PATH = Path("data/processed/agri_features.csv")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()
st.title("🚜 Predictive Maintenance Dashboard")

# Sidebar filters
machine_ids = df['machine_id'].unique()
selected_machine = st.sidebar.selectbox("Select Machine ID", machine_ids)
df_machine = df[df['machine_id'] == selected_machine]

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Summary", "📊 Failures", "🔍 Predict"])

# Tab 1: Summary
with tab1:
    st.subheader("Data Overview")
    st.write(df_machine.describe())

    st.subheader("Remaining Useful Life (RUL) Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['remaining_minutes'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("RUL (minutes)")
    st.pyplot(fig)

# Tab 2: Failure Analysis
with tab2:
    st.subheader("Failure Count by Operating Mode")
    failure_df = df[df['failure_label'] == 1]
    mode_cols = [col for col in df.columns if col.startswith("operating_mode_")]
    failure_counts = failure_df[mode_cols].sum().sort_values()

    fig2, ax2 = plt.subplots()
    failure_counts.plot(kind='barh', color='salmon', ax=ax2)
    ax2.set_title("Failures by Operating Mode")
    ax2.set_xlabel("Failure Count")
    st.pyplot(fig2)

# Tab 3: Predict from Current Sensor Readings
with tab3:
    st.subheader("Model Predictions")

    # Train quick models (cached)
    @st.cache_resource
    def train_models(df):
        X = df.drop(columns=['failure_label', 'remaining_minutes', 'machine_id'])
        y_cls = df['failure_label']
        y_reg = df['remaining_minutes']

        clf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
        reg = RandomForestRegressor(n_estimators=50, random_state=42)

        clf.fit(X, y_cls)
        reg.fit(X, y_reg)
        return clf, reg

    clf_model, reg_model = train_models(df)

    # Select one row to simulate live input
    row_idx = st.slider("Select a row index", 0, len(df_machine)-1, 0)
    input_row = df_machine.drop(columns=['failure_label', 'remaining_minutes', 'machine_id']).iloc[[row_idx]]

    # Predict
    failure_pred = clf_model.predict_proba(input_row)[0][1]
    rul_pred = reg_model.predict(input_row)[0]

    st.metric("🔧 Failure Risk Probability", f"{failure_pred:.2%}")
    st.metric("⏱️ Predicted RUL (minutes)", f"{int(rul_pred)}")
