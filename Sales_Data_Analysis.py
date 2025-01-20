import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title and Description
st.title("Sales Data Analysis App")
st.markdown("""
This app provides an analysis of sales data. 
You can upload a CSV file and view key insights, visualizations, and summaries.
""")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your sales data CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Data Summary
    st.write("### Data Summary")
    st.write(data.describe())

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    date_column = st.sidebar.selectbox("Select Date Column (if available)", options=data.columns)
    category_column = st.sidebar.selectbox("Select Category Column", options=data.columns)
    numeric_column = st.sidebar.selectbox("Select Numeric Column for Analysis", options=data.select_dtypes(include=["float64", "int64"]).columns)

    # Filter by Date Range
    if pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
        min_date, max_date = data[date_column].min(), data[date_column].max()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        if len(date_range) == 2:
            data = data[(data[date_column] >= pd.Timestamp(date_range[0])) & (data[date_column] <= pd.Timestamp(date_range[1]))]
    
    # Visualizations
    st.write("### Visualizations")

    # Bar Plot
    st.write("#### Total Sales by Category")
    if category_column in data.columns:
        sales_by_category = data.groupby(category_column)[numeric_column].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=category_column, y=numeric_column, data=sales_by_category, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Time Series Analysis
    if pd.api.types.is_datetime64_any_dtype(data[date_column]):
        st.write("#### Sales Over Time")
        sales_over_time = data.groupby(date_column)[numeric_column].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=date_column, y=numeric_column, data=sales_over_time, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Pie Chart
    st.write("#### Category Distribution")
    fig, ax = plt.subplots(figsize=(7, 7))
    data[category_column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

    # Download Processed Data
    st.write("### Download Processed Data")
    processed_csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=processed_csv, file_name="processed_sales_data.csv", mime="text/csv")

else:
    st.warning("Please upload a CSV file to proceed.")











