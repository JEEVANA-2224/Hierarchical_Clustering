import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("Wine Clustering and Prediction App")

# Load wine dataset
uploaded_file = st.file_uploader("Upload Wine CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # All features are numeric
    X = df.values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hc.fit_predict(X_scaled)
    df['Cluster'] = labels

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # Save pickles for prediction
    with open("wine_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("wine_labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    with open("wine_df.pkl", "wb") as f:
        pickle.dump(df, f)

    # Train KNN classifier for new sample prediction
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, labels)
    with open("wine_knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    # Dendrogram
    st.subheader("Hierarchical Clustering Dendrogram")
    sample_size = min(50, X_scaled.shape[0])  # for readability
    sample_idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_sample = X_scaled[sample_idx]
    Z = linkage(X_sample, method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax)
    st.pyplot(fig)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader("PCA Scatter Plot of Clusters")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_title("Wine Clusters")
    st.pyplot(fig2)

    # New wine sample prediction
    st.subheader("Predict Cluster for New Wine Sample")
    new_data = []
    for feature in df.columns[:-1]:  # skip Cluster column
        value = st.number_input(f"{feature}", value=float(df[feature].mean()), step=0.01)
        new_data.append(value)

    if st.button("Predict Cluster"):
        new_data_arr = np.array([new_data])
        new_data_scaled = scaler.transform(new_data_arr)
        predicted_cluster = knn.predict(new_data_scaled)[0]
        st.success(f"Predicted Cluster: {predicted_cluster}")

        # Show new sample in PCA scatter
        new_pca = pca.transform(new_data_scaled)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', label='Existing Wines')
        ax3.scatter(new_pca[0, 0], new_pca[0, 1], color='red', s=100, edgecolor='black', label='New Wine')
        ax3.set_xlabel("PCA 1")
        ax3.set_ylabel("PCA 2")
        ax3.set_title("Wine Cluster with New Sample")
        ax3.legend()
        st.pyplot(fig3)
