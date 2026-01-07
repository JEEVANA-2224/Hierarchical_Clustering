import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("🍷 Wine Cluster Prediction ")


# Load saved objects
with open("wine_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("wine_df.pkl", "rb") as f:
    df = pickle.load(f)

with open("wine_knn.pkl", "rb") as f:
    knn = pickle.load(f)

# Features
features = ['alcohol', 'malic_acid', 'ash', 'ash_alcanity', 'magnesium',
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue', 'od280', 'proline']

st.subheader("Enter New Wine Sample")

new_data = []
for feature in features:
    value = st.number_input(f"{feature}", value=float(df[feature].mean()), step=0.01)
    new_data.append(value)

if st.button("Predict Cluster"):
    new_data_arr = np.array([new_data])
    new_data_scaled = scaler.transform(new_data_arr)
    
    # Predict cluster
    predicted_cluster = knn.predict(new_data_scaled)[0]
    st.success(f"Predicted Cluster: {predicted_cluster}")

    # PCA for visualization
    X_scaled = scaler.transform(df[features].values)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    new_pca = pca.transform(new_data_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='tab10', label='Existing Wines')
    ax.scatter(new_pca[0, 0], new_pca[0, 1], color='red', s=100, edgecolor='black', label='New Wine')
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Wine Cluster – New Sample Highlighted")
    ax.legend()
    st.pyplot(fig)
