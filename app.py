import streamlit as st
import pickle
import numpy as np

st.title("Customer Cluster Prediction")

# Load pickled scaler and customer data
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("customer_df.pkl", "rb") as f:
    customer_df = pickle.load(f)

# Agglomerative clustering model can't be saved directly like other models,
# so we will use labels from the existing data to assign clusters
# Here we assume 3 clusters and use the nearest cluster approach
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Display info
st.write("Existing customer data with clusters")
st.dataframe(customer_df.head())

# Input new customer data
st.subheader("Enter Customer Details")
quantity = st.number_input("Total Quantity Purchased", min_value=0, value=10)
unitprice = st.number_input("Average Unit Price", min_value=0.0, value=20.0, step=0.1)

# Convert and scale
new_customer = np.array([[quantity, unitprice]])
new_customer_scaled = scaler.transform(new_customer)

# Predict cluster by nearest neighbor approach
# Compute distance to each existing scaled customer
distances = np.linalg.norm(customer_df[['Quantity', 'UnitPrice']].values - [quantity, unitprice], axis=1)
nearest_idx = np.argmin(distances)
predicted_cluster = labels[nearest_idx]

st.success(f"Predicted Cluster: {predicted_cluster}")
