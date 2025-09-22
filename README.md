# SCT_ML_2

🏬 Mall Customer Segmentation

Interactive Streamlit app for clustering mall customers using KMeans. Upload customer data, visualize clusters, predict new customer segments, and download results.

Features

Upload CSV with Annual Income (k$) and Spending Score (1-100).

Visualize clusters & centroids.

Optional Elbow Method for optimal clusters.

Live prediction for new customers.

Download clustered dataset.

Fully offline, no PyTorch/TensorFlow needed.

Usage

Train the model:

python train_model.py


Run the app:

python -m streamlit run app.py

Dataset

Required columns: Annual Income (k$) & Spending Score (1-100)

Sample dataset: Kaggle Mall Customers
