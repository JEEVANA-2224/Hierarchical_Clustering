import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np

st.title("Customer Segmentation – Hierarchical Clustering")

# Load pickled files
with open("customer_df.pkl", "rb") as f:
    customer_df = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(customer_df[['Quantity', 'UnitPrice']])

st.write("Aggregated Customer Data with Clusters", customer_df.head())

# Scatter plot
st.subheader("Customer Segmentation Scatter Plot")
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10')
ax.set_xlabel("Total Quantity (scaled)")
ax.set_ylabel("Average UnitPrice (scaled)")
ax.set_title("Customer Segmentation – Hierarchical Clustering")
st.pyplot(fig)
