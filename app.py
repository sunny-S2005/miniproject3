import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans 

st.set_page_config(page_title="Customer Personality Prediction",page_icon="🛒")
st.title("🛒Customer Personality Prediction")
st.subheader("devide customer depends on her monthly spending")

st.divider()

df= pd.read_csv("Customer.csv")

with st.expander("View Dateset"):
    df
x = df[["Age","Income","Online_Spending"]]
cluster = st.slider("Select Number Of Cluster",min_value=2,max_value=6,value=3)

model = KMeans(n_clusters=cluster,random_state=42)

df["Cluster"] = model.fit_predict(x)
cluster_name={
    0:"Budget Customer",
    1:"Regular Customer",
    2:"prenium Customer",
    3:"VIP customer",
    4:"Rich Customer",
    5:"Ultra Rich Customer"
}

df["Customer_Type"] =df["Cluster"].map(cluster_name)
with st.expander("Cluster Centre"):
    st.subheader("Cluster Centre")
    st.dataframe(df)
st.subheader("Customer Type")
st.dataframe(df[["Age","Income","Online_Spending","Cluster","Customer_Type"]])
st.subheader("Cluster centre")
centres = pd.DataFrame(model.cluster_centers_,columns=["Age","Income","Online_Spending"])
st.dataframe(centres)
st.success(f"Interia score:{model.inertia_:.2f}")
st.subheader("Graph")
fig,ax =plt.subplots(figsize=(5,3))
scatter = ax.scatter(df["Income"],df["Online_Spending"],c=df["Cluster"],cmap="ocean",s=10)
ax.scatter(model.cluster_centers_[:,1],model.cluster_centers_[:,2],marker="^",color="red",s=30,label="centroids")
ax.set_title("Customer Personality Prediction")
ax.set_xlabel("Annual Income")
ax.set_ylabel("Online Spending")
ax.grid(True)
ax.legend()
st.pyplot(fig)