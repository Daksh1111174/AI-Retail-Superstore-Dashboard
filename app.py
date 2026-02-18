import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Retail Intelligence Platform", layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.metric-card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
h1, h2, h3 {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🚀 AI Retail Intelligence Platform")
st.markdown("### Advanced Business Intelligence & Machine Learning Dashboard")

# ================= FILE UPLOAD =================
uploaded_file = st.sidebar.file_uploader("Upload Superstore CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # ================= SIDEBAR NAVIGATION =================
    page = st.sidebar.radio("Navigation",
                            ["Executive Dashboard",
                             "Sales Analytics",
                             "RFM Segmentation",
                             "Discount Analysis",
                             "Churn Prediction",
                             "Sales Forecasting"])

    # ================= EXECUTIVE DASHBOARD =================
    if page == "Executive Dashboard":

        st.subheader("📊 Executive Overview")

        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()
        total_orders = df["Order ID"].nunique()
        profit_margin = (total_profit / total_sales) * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
        col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
        col3.metric("🛒 Total Orders", total_orders)
        col4.metric("📊 Profit Margin", f"{profit_margin:.2f}%")

        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        fig = px.bar(region_sales, x="Region", y="Sales", title="Region-wise Sales")
        st.plotly_chart(fig, use_container_width=True)

    # ================= SALES ANALYTICS =================
    elif page == "Sales Analytics":

        st.subheader("📈 Sales Trends")

        monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
        monthly["Order Date"] = monthly["Order Date"].astype(str)

        fig = px.line(monthly, x="Order Date", y="Sales",
                      markers=True, title="Monthly Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

        category_sales = df.groupby("Category")["Profit"].sum().reset_index()
        fig2 = px.pie(category_sales, names="Category", values="Profit",
                      title="Profit by Category")
        st.plotly_chart(fig2, use_container_width=True)

    # ================= RFM =================
    elif page == "RFM Segmentation":

        st.subheader("👥 Customer Segmentation (RFM)")

        snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("Customer Name").agg({
            "Order Date": lambda x: (snapshot_date - x.max()).days,
            "Order ID": "count",
            "Sales": "sum"
        })

        rfm.columns = ["Recency", "Frequency", "Monetary"]

        rfm["Segment"] = pd.qcut(rfm["Monetary"], 4,
                                 labels=["Low Value", "Medium", "High", "Premium"])

        segment_counts = rfm["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]

        fig = px.bar(segment_counts, x="Segment", y="Count",
                     title="Customer Segments")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rfm.head())

    # ================= DISCOUNT =================
    elif page == "Discount Analysis":

        st.subheader("🏷 Discount Impact Analysis")

        fig = px.scatter(df, x="Discount", y="Profit",
                         size="Sales", color="Category",
                         title="Discount vs Profit Impact")
        st.plotly_chart(fig, use_container_width=True)

        correlation = df["Discount"].corr(df["Profit"])
        st.write(f"### Correlation between Discount & Profit: {correlation:.2f}")

    # ================= CHURN =================
    elif page == "Churn Prediction":

        st.subheader("🤖 Machine Learning - Customer Churn")

        snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

        customer_df = df.groupby("Customer Name").agg({
            "Order Date": lambda x: (snapshot_date - x.max()).days,
            "Order ID": "count",
            "Sales": "sum",
            "Profit": "sum"
        }).reset_index()

        customer_df.columns = ["Customer Name", "Recency", "Frequency", "Monetary", "Profit"]

        customer_df["Churn"] = np.where(customer_df["Recency"] > 90, 1, 0)

        X = customer_df[["Recency", "Frequency", "Monetary", "Profit"]]
        y = customer_df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        st.write(f"### Model Accuracy: {accuracy*100:.2f}%")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        fig = px.bar(importance, x="Feature", y="Importance",
                     title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    # ================= FORECASTING =================
    elif page == "Sales Forecasting":

        st.subheader("🔮 90-Day Sales Forecast")

        forecast_df = df.groupby("Order Date")["Sales"].sum().reset_index()
        forecast_df = forecast_df.rename(columns={"Order Date": "ds", "Sales": "y"})

        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        fig = px.line(forecast, x="ds", y="yhat",
                      title="Forecasted Sales")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬅ Upload your Superstore dataset to begin.")
