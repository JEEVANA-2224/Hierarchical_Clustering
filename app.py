import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Customer Cluster Prediction for New Data")

# Load pickled files
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("customer_df.pkl", "rb") as f:
    customer_df = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Display info
st.subheader("Existing Customer Clusters")
st.dataframe(customer_df.head())

# Input new customer details
st.subheader("Enter New Customer Details")
quantity = st.number_input("Total Quantity Purchased", min_value=0, value=10)
unitprice = st.number_input("Average Unit Price", min_value=0.0, value=20.0, step=0.1)

# Scale new data
new_customer = np.array([[quantity, unitprice]])
new_customer_scaled = scaler.transform(new_customer)

# Nearest neighbor assignment
X_scaled = scaler.transform(customer_df[['Quantity', 'UnitPrice']].values)
distances = np.linalg.norm(X_scaled - new_customer_scaled, axis=1)
nearest_idx = np.argmin(distances)
predicted_cluster = labels[nearest_idx]

st.success(f"Predicted Cluster for the new customer: {predicted_cluster}")

# Optional: scatter plot showing new customer
st.subheader("Cluster Visualization")
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    X_scaled[:, 0], X_scaled[:, 1],
    c=labels, cmap='tab10', label='Existing Customers'
)
ax.scatter(
    new_customer_scaled[0, 0],
    new_customer_scaled[0, 1],
    color='red', s=100, label='New Customer', edgecolor='black'
)
ax.set_xlabel("Total Quantity (scaled)")
ax.set_ylabel("Average UnitPrice (scaled)")
ax.set_title("Customer Segmentation – Hierarchical Clustering")
ax.legend()
st.pyplot(fig)
