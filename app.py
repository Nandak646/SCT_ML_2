import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---- Page Configuration ----
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide", page_icon="🏬")
st.title("🏬 Mall Customer Segmentation with Saved Model")

# ---- Sidebar ----
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
show_elbow = st.sidebar.checkbox("Show Elbow Method")

# ---- Load saved model and scaler ----
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---- Main App ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---- Features ----
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # ---- Standardize Data ----
    X_scaled = scaler.transform(X)

    # ---- Optional: Elbow Method ----
    if show_elbow:
        wcss = []
        for i in range(1, 11):
            k = KMeans(n_clusters=i, init='k-means++', random_state=42)
            k.fit(X_scaled)
            wcss.append(k.inertia_)
        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(range(1,11), wcss, marker='o', linestyle='--', color='blue')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

    # ---- Assign clusters ----
    df['Cluster'] = kmeans.predict(X_scaled)

    # ---- Visualization ----
    st.subheader("Customer Segments")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x=features[0], y=features[1],
        hue='Cluster', palette='Set1', data=df, s=100, ax=ax2
    )
    # Centroids
    ax2.scatter(
        kmeans.cluster_centers_[:,0]*X[features[0]].std() + X[features[0]].mean(),
        kmeans.cluster_centers_[:,1]*X[features[1]].std() + X[features[1]].mean(),
        s=300, c='black', marker='X', label='Centroids'
    )
    ax2.set_title("Customer Segments (KMeans)")
    ax2.legend()
    st.pyplot(fig2)

    # ---- Download Clustered Dataset ----
    st.subheader("Download Clustered Dataset")
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "mall_customers_clustered.csv", "text/csv")

    # ---- Live Prediction for New Customer ----
    st.subheader("🔮 Predict Cluster for New Customer")
    with st.form("prediction_form"):
        input_data = []
        for feature in features:
            value = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
            input_data.append(value)
        submitted = st.form_submit_button("Predict Cluster")
        if submitted:
            input_scaled = scaler.transform([input_data])
            cluster_pred = kmeans.predict(input_scaled)[0]
            st.success(f"Predicted Cluster: {cluster_pred}")

else:
    st.info("ℹ️ Upload a CSV file to get started.")
