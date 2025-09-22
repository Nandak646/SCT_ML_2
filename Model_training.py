# train_model.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# ---- Load Dataset ----
df = pd.read_csv("C:/Users/nanda/OneDrive/Documents/Nanda's ML Tasks/K means clustering/Mall_Customers.csv")

# ---- Features for Clustering ----
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# ---- Standardize Data ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Train KMeans ----
n_clusters = 5  # you can choose optimal clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# ---- Save Model and Scaler ----
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved as 'kmeans_model.pkl' and 'scaler.pkl'")

# ---- Assign Clusters to Dataset ----
df['Cluster'] = kmeans.predict(X_scaled)
df.to_csv("mall_customers_clustered.csv", index=False)
print("✅ Clustered dataset saved as 'mall_customers_clustered.csv'")
