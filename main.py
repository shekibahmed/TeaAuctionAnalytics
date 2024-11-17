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

# Create charts for each centre
centres = sorted(df['Centre'].unique())
num_centres = len(centres)
rows = (num_centres + 1) // 2  # Calculate number of rows needed (2 charts per row)

# Create subplot grid
fig = make_subplots(
    rows=rows, 
    cols=2,
    specs=[[{"secondary_y": True}] * 2] * rows,
    subplot_titles=[f"{centre} Leaf Trends" for centre in centres]
)

# Create charts for each centre
for idx, centre in enumerate(centres):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    centre_df = df[df['Centre'] == centre]
    
    # Bar charts for Sold and Unsold
    fig.add_trace(
        go.Bar(
            name='Sold Qty (Ton)', 
            x=centre_df['Sale No'], 
            y=centre_df['Sold Qty (Ton)'],
            marker_color='#FF9966',
            showlegend=(idx == 0)  # Show legend only for first centre
        ),
        row=row, col=col,
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            name='Unsold Qty (Ton)', 
            x=centre_df['Sale No'], 
            y=centre_df['Unsold Qty (Ton)'],
            marker_color='#808080',
            showlegend=(idx == 0)
        ),
        row=row, col=col,
        secondary_y=False
    )
    
    # Line chart for Sales Price
    fig.add_trace(
        go.Scatter(
            name='Sales Price (Kg)', 
            x=centre_df['Sale No'], 
            y=centre_df['Sales Price(Kg)'],
            line=dict(color='#3366CC', width=2),
            showlegend=(idx == 0)
        ),
        row=row, col=col,
        secondary_y=True
    )

# Update layout
fig.update_layout(
    height=400 * rows,
    width=1200,
    barmode='group',
    hovermode='x unified',
    template='plotly_white',
    margin=dict(t=50, b=20, l=60, r=60)
)

# Update axes labels for all subplots
for i in range(1, rows + 1):
    for j in range(1, 3):
        if (i-1)*2 + j <= num_centres:  # Only update if subplot exists
            fig.update_xaxes(title_text='Sale No', row=i, col=j)
            fig.update_yaxes(title_text='Quantity (Tons)', secondary_y=False, row=i, col=j)
            fig.update_yaxes(title_text='Price (₹/Kg)', secondary_y=True, row=i, col=j)

st.plotly_chart(fig, use_container_width=True)

# Summary Metrics
st.subheader("Key Metrics by Centre")

# Create metrics for each centre
metrics_cols = st.columns(len(centres))

for idx, (col, centre) in enumerate(zip(metrics_cols, centres)):
    centre_df = df[df['Centre'] == centre]
    
    with col:
        st.markdown(f"**{centre}**")
        st.metric(
            "Avg Sales Price",
            f"₹{centre_df['Sales Price(Kg)'].mean():.2f}/Kg",
            f"{centre_df['Sales Price(Kg)'].pct_change().mean()*100:.1f}%"
        )
        st.metric(
            "Total Sold Qty",
            f"{centre_df['Sold Qty (Ton)'].sum():,.0f} Tons",
            f"{centre_df['Sold Qty (Ton)'].pct_change().mean()*100:.1f}%"
        )
        st.metric(
            "Market Efficiency",
            f"{(centre_df['Sold Qty (Ton)'].sum() / (centre_df['Sold Qty (Ton)'].sum() + centre_df['Unsold Qty (Ton)'].sum()) * 100):.1f}%"
        )

# Automated Insights
st.subheader("Market Insights")
insights = generate_insights(df)
for insight in insights:
    st.info(insight)

# Data Table
st.subheader("Detailed Data View")
st.dataframe(df.style.format({
    'Sales Price(Kg)': '₹{:.2f}',
    'Sold Qty (Ton)': '{:,.0f}',
    'Unsold Qty (Ton)': '{:,.0f}'
}), use_container_width=True)
