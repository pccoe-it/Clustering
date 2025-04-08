import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Traffic Congestion Clustering", layout="wide")

st.title("Traffic Congestion Clustering and Insights with K-Means")
st.markdown("""
This app uses K-Means clustering to analyze traffic data.
It splits the data into training and testing sets, trains a K-Means model, and then predicts clusters for unseen (testing) data.
Finally, it provides visualizations and insights into the characteristics of each cluster.
""")

# -----------------------------
# 1. Data Loading
# -----------------------------
uploaded_file = st.file_uploader("Upload your traffic CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    # Check required columns
    required_columns = [
        "Traffic Volume", 
        "Vehicle Speed (km/h)", 
        "Traffic Density (vehicles/km)", 
        "Accidents", 
        "Weather Conditions", 
        "Road Type",
        "Congestion Level"  # Used for evaluation (optional)
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # -----------------------------
        # 2. Data Cleaning & Partitioning
        # -----------------------------
        df.dropna(inplace=True)
        features = ["Traffic Volume", "Vehicle Speed (km/h)", "Traffic Density (vehicles/km)", 
                    "Accidents", "Weather Conditions", "Road Type"]
        target = "Congestion Level"  # Provided true labels for optional evaluation
        
        # Split data into training and testing sets (70% / 30%)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])
        st.markdown(f"**Training Set:** {train_df.shape} rows &nbsp;&nbsp;&nbsp; **Test Set:** {test_df.shape} rows")
        
        # -----------------------------
        # 3. Preprocessing
        # -----------------------------
        num_cols = ["Traffic Volume", "Vehicle Speed (km/h)", "Traffic Density (vehicles/km)", "Accidents"]
        cat_cols = ["Weather Conditions", "Road Type"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(), cat_cols)
            ]
        )
        X_train = preprocessor.fit_transform(train_df[features])
        X_test = preprocessor.transform(test_df[features])
        
        # -----------------------------
        # 4. K-Means Clustering Model Training
        # -----------------------------
        # Let the user select the number of clusters.
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=200, value=153)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_train)
        
        # Cluster predictions on both train and test data
        train_clusters = kmeans.predict(X_train)
        test_clusters = kmeans.predict(X_test)
        
        # Attach clusters to dataframes
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["Cluster"] = train_clusters
        test_df["Cluster"] = test_clusters
        
        # -----------------------------
        # 5. Evaluation (Internal Metrics)
        # -----------------------------
        silhouette = silhouette_score(X_train, train_clusters)
        ch_score = calinski_harabasz_score(X_train, train_clusters)
        db_score = davies_bouldin_score(X_train, train_clusters)
        
        st.subheader("Clustering Evaluation on Training Data")
        st.write(f"**Silhouette Score:** {silhouette:.3f}")
        st.write(f"**Calinski-Harabasz Score:** {ch_score:.3f}")
        st.write(f"**Davies-Bouldin Score:** {db_score:.3f}")
        
        # -----------------------------
        # 6. Visualization: PCA Plot of Clusters
        # -----------------------------
        st.subheader("PCA Visualization of Clusters (Training Data)")
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        pca_df = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = train_clusters
        
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="viridis", ax=ax)
        ax.set_title("Clusters Visualization Using PCA")
        st.pyplot(fig)
        
        # -----------------------------
        # 7. Insights from Testing Data
        # -----------------------------
        st.subheader("Insights on Testing Data")
        
        # Display a sample of cluster assignments
        st.write("**Sample of Test Data with Cluster Assignments:**")
        st.write(test_df[["Traffic Volume", "Vehicle Speed (km/h)", "Traffic Density (vehicles/km)", "Accidents", "Cluster"]].head(20))
        
        # Aggregate insights: Average metrics per cluster for test data
        insight_cols = ["Traffic Volume", "Vehicle Speed (km/h)", "Traffic Density (vehicles/km)", "Accidents"]
        cluster_insights = test_df.groupby("Cluster")[insight_cols].mean().reset_index()
        st.write("**Average Values per Cluster:**")
        st.dataframe(cluster_insights)
        
        # Plot the distribution of clusters on test data
        st.subheader("Cluster Distribution on Test Data")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        cluster_counts = test_df["Cluster"].value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax2)
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Number of Instances")
        ax2.set_title("Test Data Cluster Distribution")
        st.pyplot(fig2)
        
        # Optional: Compare predicted clusters to true congestion levels
        st.subheader("Comparison: Predicted Clusters vs. True Congestion Levels")
        comparison = test_df[["Cluster", target]].head(20)
        st.write(comparison)
        
        st.markdown("""
        **Interpretation & Next Steps:**
        - The average values per cluster can help you label clusters (e.g., a cluster with high traffic volume and low speed might represent heavy congestion).
        - The cluster distribution shows how frequently each traffic pattern occurs.
        - Comparing clusters with true congestion levels (if available) can indicate where the unsupervised model aligns with known patterns.
        - Use these insights to fine-tune features or adjust the number of clusters for better real-world performance.
        """)

        # -----------------------------
        # 8. Real-time Test Input & Cluster Prediction
        # -----------------------------
        st.header("🚗 Real-Time Traffic Prediction")

        st.markdown("Enter traffic parameters below to predict the congestion cluster and its interpretation.")

        with st.form("predict_form"):
            volume = st.number_input("Traffic Volume", min_value=0, value=500)
            speed = st.number_input("Vehicle Speed (km/h)", min_value=0, value=40)
            density = st.number_input("Traffic Density (vehicles/km)", min_value=0.0, value=20.0)
            accidents = st.number_input("Number of Accidents", min_value=0, value=0)
            weather = st.selectbox("Weather Condition", df["Weather Conditions"].unique())
            road_type = st.selectbox("Road Type", df["Road Type"].unique())
            
            submit = st.form_submit_button("Predict Congestion Level")

        if submit:
            input_df = pd.DataFrame([{
                "Traffic Volume": volume,
                "Vehicle Speed (km/h)": speed,
                "Traffic Density (vehicles/km)": density,
                "Accidents": accidents,
                "Weather Conditions": weather,
                "Road Type": road_type
            }])

            input_processed = preprocessor.transform(input_df)
            cluster_pred = kmeans.predict(input_processed)[0]

            # Example interpretation logic — this should be based on your cluster insights
            interpretation = {
                0: "🟢 Free Flowing",
                1: "🟡 Moderate Congestion",
                2: "🔴 Heavy Congestion",
                3: "🟠 Variable Flow"
            }
            st.subheader("Prediction Results")
            st.write(f"**Predicted Cluster:** {cluster_pred}")
            st.write(f"**Congestion Level:** {interpretation.get(cluster_pred, 'Unknown Cluster – Please interpret based on cluster stats')}")

            st.markdown("💡 **Note:** These interpretations are based on average cluster characteristics observed during testing. You can update the mapping logic after reviewing the real data.")

else:
    st.info("Awaiting CSV file upload...")
