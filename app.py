import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Set the title and description
st.title("Traffic Congestion Clustering with K-Means")
st.write("""
This application uses K-Means clustering to group traffic segments based on traffic volume, speed, density,
accidents, and environmental conditions (weather and road type). It then provides evaluation metrics and visualizations.
""")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload your traffic CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    # Check if all required columns exist
    required_columns = [
        "Traffic Volume", 
        "Vehicle Speed (km/h)", 
        "Traffic Density (vehicles/km)", 
        "Accidents", 
        "Weather Conditions", 
        "Road Type",
        "Congestion Level"  # Optional: Only needed if you want to compare with true labels.
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # ============================
        # Preprocessing
        # ============================
        # We use all columns except the true congestion level for clustering.
        features = [
            "Traffic Volume", 
            "Vehicle Speed (km/h)", 
            "Traffic Density (vehicles/km)", 
            "Accidents", 
            "Weather Conditions", 
            "Road Type"
        ]
        
        # Define which features are numerical and which are categorical.
        num_cols = ["Traffic Volume", "Vehicle Speed (km/h)", "Traffic Density (vehicles/km)", "Accidents"]
        cat_cols = ["Weather Conditions", "Road Type"]
        
        # Create a preprocessing pipeline for scaling numbers and encoding categorical features.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(), cat_cols)
            ]
        )
        
        # Transform features
        X_processed = preprocessor.fit_transform(df[features])
        
        # ============================
        # K-Means Clustering
        # ============================
        # Allow user to adjust the number of clusters.
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)
        
        # Train the K-Means model.
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_processed)
        
        # ============================
        # Evaluation Metrics
        # ============================
        silhouette = silhouette_score(X_processed, df["Cluster"])
        ch_score = calinski_harabasz_score(X_processed, df["Cluster"])
        db_score = davies_bouldin_score(X_processed, df["Cluster"])
        
        st.subheader("Clustering Evaluation Metrics")
        st.write(f"**Silhouette Score:** {silhouette:.3f}")
        st.write(f"**Calinski-Harabasz Score:** {ch_score:.3f}")
        st.write(f"**Davies-Bouldin Score:** {db_score:.3f}")
        
        # ============================
        # Display Cluster Assignments
        # ============================
        st.subheader("Sample Cluster Assignments")
        st.write(df[["Traffic Volume", "Vehicle Speed (km/h)", "Traffic Density (vehicles/km)", "Cluster"]].head(20))
        
        # ============================
        # Visualize Clusters Using PCA
        # ============================
        st.subheader("PCA Visualization of Clusters")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)
        df["PC1"] = X_pca[:, 0]
        df["PC2"] = X_pca[:, 1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="viridis", ax=ax)
        ax.set_title("Clusters Visualization (via PCA)")
        st.pyplot(fig)
        
        # Optional: Compare clusters to true congestion levels (if provided)
        if "Congestion Level" in df.columns:
            st.subheader("Comparison: Cluster vs. True Congestion Level")
            comparison_df = df[["Cluster", "Congestion Level"]].copy()
            st.write(comparison_df.head(20))
            
            # Display distribution of clusters
            st.subheader("Cluster Distribution")
            st.bar_chart(df["Cluster"].value_counts())
else:
    st.info("Awaiting CSV file upload...")
