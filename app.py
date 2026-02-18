import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="AI Retail Superstore Dashboard", layout="wide")

st.title("📊 AI-Powered Retail Sales Forecasting & Profit Optimization")

# File uploader
uploaded_file = st.file_uploader("Upload Superstore CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    # Convert dates
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Sidebar Filters
    st.sidebar.header("🔎 Filters")
    region = st.sidebar.multiselect("Select Region", df["Region"].unique(), default=df["Region"].unique())
    category = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())

    filtered_df = df[(df["Region"].isin(region)) & (df["Category"].isin(category))]

    # ================= KPI SECTION =================
    st.subheader("📌 Key Performance Indicators")

    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Profit"].sum()
    total_quantity = filtered_df["Quantity"].sum()
    profit_margin = (total_profit / total_sales) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Total Profit", f"${total_profit:,.2f}")
    col3.metric("Total Quantity Sold", total_quantity)
    col4.metric("Profit Margin %", f"{profit_margin:.2f}%")

    # ================= SALES TREND =================
    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = filtered_df.groupby(filtered_df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)

    fig1 = px.line(monthly_sales, x="Order Date", y="Sales", title="Monthly Sales")
    st.plotly_chart(fig1, use_container_width=True)

    # ================= CATEGORY ANALYSIS =================
    st.subheader("📊 Category-wise Sales")

    category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig2 = px.bar(category_sales, x="Category", y="Sales", color="Category")
    st.plotly_chart(fig2, use_container_width=True)

    # ================= PROFIT OPTIMIZATION =================
    st.subheader("💰 Top & Loss Making Products")

    product_profit = filtered_df.groupby("Product Name")["Profit"].sum().reset_index()

    top_products = product_profit.sort_values(by="Profit", ascending=False).head(10)
    loss_products = product_profit.sort_values(by="Profit").head(10)

    st.write("### 🔝 Top Profitable Products")
    st.dataframe(top_products)

    st.write("### ⚠ Loss Making Products")
    st.dataframe(loss_products)

    # ================= FORECASTING =================
    st.subheader("🔮 Sales Forecast (Next 90 Days)")

    forecast_df = df.groupby("Order Date")["Sales"].sum().reset_index()
    forecast_df = forecast_df.rename(columns={"Order Date": "ds", "Sales": "y"})

    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    fig3 = px.line(forecast, x="ds", y="yhat", title="Sales Forecast")
    st.plotly_chart(fig3, use_container_width=True)

    st.success("Dashboard Loaded Successfully ✅")
