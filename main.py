import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import process_excel_data, generate_insights
from styles import apply_custom_styles

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")
apply_custom_styles()

st.title("Sales Analytics Dashboard")

# File upload section
uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls', 'csv'])

# Load default or uploaded data
if uploaded_file is not None:
    try:
        df = process_excel_data(uploaded_file)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        df = pd.read_csv('assets/default_data.csv')
else:
    df = pd.read_csv('assets/default_data.csv')

# Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Main Trend Chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar charts for Sold and Unsold
    fig.add_trace(
        go.Bar(name='Sold Qty (Ton)', x=df['Sale No'], y=df['Sold Qty (Ton)'],
               marker_color='#FF9966'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(name='Unsold Qty (Ton)', x=df['Sale No'], y=df['Unsold Qty (Ton)'],
               marker_color='#808080'),
        secondary_y=False,
    )
    
    # Line chart for Sales Price
    fig.add_trace(
        go.Scatter(name='Sales Price (₹)', x=df['Sale No'], y=df['Sales Price (₹)'],
                  line=dict(color='#3366CC', width=2)),
        secondary_y=True,
    )
    
    fig.update_layout(
        title='North India CTC Leaf Trends',
        barmode='group',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    fig.update_xaxes(title_text='Sale No')
    fig.update_yaxes(title_text='Quantity (Tons)', secondary_y=False)
    fig.update_yaxes(title_text='Price (₹)', secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Summary Metrics
    st.subheader("Key Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric(
            "Avg Sales Price",
            f"₹{df['Sales Price (₹)'].mean():.2f}",
            f"{df['Sales Price (₹)'].pct_change().mean()*100:.1f}%"
        )
        st.metric(
            "Total Sold Qty",
            f"{df['Sold Qty (Ton)'].sum():,.0f} Tons",
            f"{df['Sold Qty (Ton)'].pct_change().mean()*100:.1f}%"
        )
    
    with metrics_col2:
        st.metric(
            "Avg Sold Qty",
            f"{df['Sold Qty (Ton)'].mean():,.0f} Tons",
            f"{df['Sold Qty (Ton)'].pct_change().mean()*100:.1f}%"
        )
        st.metric(
            "Total Unsold Qty",
            f"{df['Unsold Qty (Ton)'].sum():,.0f} Tons",
            f"{df['Unsold Qty (Ton)'].pct_change().mean()*100:.1f}%"
        )

# Automated Insights
st.subheader("Market Insights")
insights = generate_insights(df)
for insight in insights:
    st.info(insight)

# Data Table
st.subheader("Detailed Data View")
st.dataframe(df.style.format({
    'Sales Price (₹)': '₹{:.2f}',
    'Sold Qty (Ton)': '{:,.0f}',
    'Unsold Qty (Ton)': '{:,.0f}'
}), use_container_width=True)
